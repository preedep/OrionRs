use anyhow::Result;

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
        let query_vector = self.embedding_service
            .generate_embedding(&query.text)
            .await?;
        
        let results = self.vector_db
            .search(query_vector, query.limit, query.score_threshold)
            .await?;
        
        Ok(results)
    }
}
