mod application;
mod config;
mod domain;
mod infrastructure;

use anyhow::Result;
use application::{RagService, SearchService};
use config::Config;
use domain::SearchQuery;
use infrastructure::{EmbeddingService, OllamaClient, QdrantClient};
use log::info;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    pretty_env_logger::init();
    
    info!("Starting OrionRs Vector Search Application");

    let config = Config::from_env()?;
    info!("Configuration loaded successfully");
    
    let mode = env::args().nth(1).unwrap_or_else(|| "search".to_string());

    println!("🔍 Vector Search with Score - Qdrant Client");
    println!("   Host: {}:{}", config.qdrant_host, config.qdrant_port);
    println!("   Collection: {}", config.collection_name);
    println!();

    let embedding_service = EmbeddingService::new();
    let vector_db = QdrantClient::new(
        &config.qdrant_host,
        config.qdrant_port,
        &config.collection_name,
    )?;

    match mode.as_str() {
        "rag" => {
            println!("🤖 RAG Mode - Using Ollama: {}", config.ollama_base_url);
            println!("   Model: {}", config.ollama_model);
            println!();
            
            let ollama_client = OllamaClient::new(&config.ollama_base_url)?;
            let rag_service = RagService::new(embedding_service, vector_db, ollama_client);

            let question = env::args()
                .nth(2)
                .unwrap_or_else(|| "What is the balance information?".to_string());

            println!("❓ Question: {}", question);
            println!();

            let response = rag_service
                .ask(&question, &config.ollama_model, 5)
                .await?;

            println!("✅ Answer:");
            println!("{}", "=".repeat(80));
            println!("{}", response.answer);
            println!();
            println!("📚 Sources ({} documents):", response.sources.len());
            println!("{}", "=".repeat(80));

            for (idx, source) in response.sources.iter().enumerate() {
                println!();
                println!("{}. Score: {:.4}", idx + 1, source.score);
                println!("   Source: {}", source.source_file);
                println!(
                    "   Text: {}...",
                    source.text.chars().take(100).collect::<String>()
                );
                println!("{}", "-".repeat(80));
            }
        }
        _ => {
            println!("🔎 Search Mode");
            
            let search_service = SearchService::new(embedding_service, vector_db);

            let query = SearchQuery::new("balance", 10);

            println!("🔎 Searching for: '{}'", query.text);
            println!();

            let results = search_service.search(query).await?;

            println!("✅ Found {} results:", results.len());
            println!("{}", "=".repeat(80));

            for (idx, result) in results.iter().enumerate() {
                println!();
                println!("{}. Score: {:.4}", idx + 1, result.score);
                println!("   ID: {}", result.id);
                println!("   Collection: {}", result.collection);
                println!("   Source: {}", result.source_file);
                println!(
                    "   Text: {}...",
                    result.text.chars().take(150).collect::<String>()
                );
                println!("{}", "-".repeat(80));
            }
        }
    }

    Ok(())
}
