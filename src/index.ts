import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { type Document } from "@langchain/core/documents";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { CONFIG } from "./config.ts";
import { DocumentProcessor } from "./documentProcessor.ts";
import { type PretrainedOptions } from "@huggingface/transformers";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";

let _neo4jVectorStore = null

async function clearAll(vectorStore: Neo4jVectorStore, nodeLabel: string): Promise<void> {
    console.log("🗑️  Removendo todos os documentos existentes...");
    await vectorStore.query(
        `MATCH (n:\`${nodeLabel}\`) DETACH DELETE n`
    )
    console.log("✅ Documentos removidos com sucesso\n");
}


try {
    console.log("🚀 Inicializando sistema de Embeddings com Neo4j...\n");

    const documentProcessor = new DocumentProcessor(
        CONFIG.pdf.path,
        CONFIG.textSplitter,
    )
    const documents = await documentProcessor.loadAndSplit()
    const embeddings = new HuggingFaceTransformersEmbeddings({
        model: CONFIG.embedding.modelName,
        pretrainedOptions: CONFIG.embedding.pretrainedOptions as PretrainedOptions
    })
    // const response = await embeddings.embedQuery(
    //     "JavaScript"
    // )
    // const response = await embeddings.embedDocuments([
    //     "JavaScript"
    // ])
    // console.log('response', response)

    _neo4jVectorStore = await Neo4jVectorStore.fromExistingGraph(
        embeddings,
        CONFIG.neo4j
    )

    clearAll(_neo4jVectorStore, CONFIG.neo4j.nodeLabel)
    for (const [index, doc] of documents.entries()) {
        console.log(`✅ Adicionando documento ${index + 1}/${documents.length}`);
        await _neo4jVectorStore.addDocuments([doc])
    }
    console.log("\n✅ Base de dados populada com sucesso!\n");


    // ==================== STEP 2: SERVER WEB ====================
    console.log("🌍 ETAPA 2: Iniciando servidor web com API de busca...\n");

    // Configuração para resolver diretorios no ESM
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = dirname(__filename);

    const server = createServer(async (req, res) => {
        try {
            res.setHeader("Access-Control-Allow-Origin", "*");

            // Servir a interface visual criada no HTML
            if (req.method === 'GET' && (req.url === '/' || req.url === '/index.html')) {
                const indexPath = join(__dirname, '..', 'index.html');
                const content = await readFile(indexPath, 'utf-8');
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(content);
                return;
            }

            // Endpoint da API de busca de Embeddings no Neo4j
            const url = new URL(req.url || "/", `http://${req.headers.host}`);
            if (req.method === 'GET' && url.pathname === '/api/search') {
                const q = url.searchParams.get('q');
                if (!q) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: "Parâmetro 'q' é obrigatório" }));
                    return;
                }

                console.log(`[API] Buscando por: "${q}"`);
                // Consultamos o banco Neo4j incluindo o score no resultado
                const searchResults = await _neo4jVectorStore.similaritySearchWithScore(
                    q,
                    CONFIG.similarity.topK
                );

                // Formando o array mapeando os docs e injetando o score nos metadados 
                const results = searchResults.map(([doc, score]: [Document, number]) => {
                    return {
                        ...doc,
                        metadata: {
                            ...doc.metadata,
                            score: score
                        }
                    }
                })

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ results }));
                return;
            }

            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end("Not Found");
        } catch (err) {
            console.error("[HTTP Error]", err);
            res.writeHead(500, { 'Content-Type': 'text/plain' });
            res.end("Internal Server Error");
        }
    });

    server.listen(3001, () => {
        console.log(`\n🚀 O Servidor backend e a camada visual estão rodando em:`);
        console.log(`👉 http://localhost:3001/`);
        console.log(`(Pressione Ctrl + C para parar o servidor)`);
    });

} catch (error) {
    console.error('error', error)
} finally {
    // Comentamos o close() abaixo para permitir que a conexao ao Neo4j 
    // fique viva escutando as requisicoes HTTP
    // await _neo4jVectorStore?.close();
}