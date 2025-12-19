mod application;
mod config;
mod domain;
mod infrastructure;

use anyhow::Result;
use application::SearchService;
use config::Config;
use domain::SearchQuery;
use infrastructure::{EmbeddingService, QdrantClient};

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::from_env()?;

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

    Ok(())
}
