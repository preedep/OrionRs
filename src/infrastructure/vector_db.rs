use anyhow::Result;
use log::{debug, info, warn};
use serde::Deserialize;
use std::collections::HashMap;

use crate::domain::SearchResult;

pub struct QdrantClient {
    http_client: reqwest::Client,
    base_url: String,
    collection_name: String,
}

#[derive(Deserialize)]
struct QdrantSearchResponse {
    result: Vec<QdrantScoredPoint>,
}

#[derive(Deserialize)]
struct QdrantScoredPoint {
    id: u64,
    score: f32,
    payload: HashMap<String, serde_json::Value>,
}

impl QdrantClient {
    pub fn new(host: &str, port: u16, collection_name: &str) -> Result<Self> {
        info!("Initializing Qdrant client");
        debug!("Qdrant URL: http://{}:{}", host, port);
        debug!("Collection: {}", collection_name);
        
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        
        info!("Qdrant client initialized successfully");
        
        Ok(Self {
            http_client,
            base_url: format!("http://{}:{}", host, port),
            collection_name: collection_name.to_string(),
        })
    }

    pub async fn search(
        &self,
        query_vector: Vec<f32>,
        limit: u64,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        info!("Searching in Qdrant collection: {}", self.collection_name);
        debug!("Search parameters - limit: {}, score_threshold: {:?}, vector_dim: {}", 
               limit, score_threshold, query_vector.len());
        
        let url = format!("{}/collections/{}/points/search", self.base_url, self.collection_name);
        
        let mut body = serde_json::json!({
            "vector": query_vector,
            "limit": limit,
            "with_payload": true,
        });
        
        if let Some(threshold) = score_threshold {
            body["score_threshold"] = serde_json::json!(threshold);
            debug!("Using score threshold: {}", threshold);
        }
        
        let start = std::time::Instant::now();
        let response = self.http_client
            .post(&url)
            .json(&body)
            .send()
            .await?;
        
        let request_duration = start.elapsed();
        debug!("Qdrant request completed in {:.2}ms", request_duration.as_secs_f64() * 1000.0);
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            warn!("Qdrant search failed with status {}: {}", status, error_text);
            anyhow::bail!("Qdrant search failed ({}): {}", status, error_text);
        }
        
        let search_response: QdrantSearchResponse = response.json().await?;
        
        let results = search_response
            .result
            .into_iter()
            .map(|point| self.parse_scored_point(point))
            .collect::<Result<Vec<_>>>()?;

        info!("Found {} results from Qdrant", results.len());
        if !results.is_empty() {
            debug!("Top result score: {:.4}", results[0].score);
        }

        Ok(results)
    }

    fn parse_scored_point(&self, point: QdrantScoredPoint) -> Result<SearchResult> {
        let text = point.payload
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default();
        
        let source_file = point.payload
            .get("source_file")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        
        let collection = point.payload
            .get("collection")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.collection_name.clone());

        Ok(SearchResult {
            id: point.id,
            score: point.score,
            text,
            source_file,
            collection,
        })
    }
}
