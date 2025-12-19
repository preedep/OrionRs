use anyhow::Result;
use log::{debug, info};

use crate::domain::{SearchQuery, SearchResult};
use crate::infrastructure::{EmbeddingService, QdrantClient};

pub struct SearchService {
    embedding_service: EmbeddingService,
    vector_db: QdrantClient,
}

impl SearchService {
    pub fn new(embedding_service: EmbeddingService, vector_db: QdrantClient) -> Self {
        Self {
            embedding_service,
            vector_db,
        }
    }

    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        info!("Starting vector search");
        debug!("Query: {}", query.text);
        debug!("Limit: {}, Threshold: {:?}", query.limit, query.score_threshold);
        
        let query_vector = self.embedding_service
            .generate_embedding(&query.text)
            .await?;
        
        let results = self.vector_db
            .search(query_vector, query.limit, query.score_threshold)
            .await?;
        
        info!("Search completed successfully with {} results", results.len());
        
        Ok(results)
    }
}
