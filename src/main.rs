use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Deserialize)]
struct SearchResult {
    id: u64,
    score: f32,
    text: String,
    source_file: String,
    collection: String,
}

struct VectorSearchClient {
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

impl VectorSearchClient {
    async fn new(host: &str, port: u16, collection_name: &str) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        
        Ok(Self {
            http_client,
            base_url: format!("http://{}:{}", host, port),
            collection_name: collection_name.to_string(),
        })
    }

    async fn search_with_score(
        &self,
        query_vector: Vec<f32>,
        limit: u64,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        let url = format!("{}/collections/{}/points/search", self.base_url, self.collection_name);
        
        let mut body = serde_json::json!({
            "vector": query_vector,
            "limit": limit,
            "with_payload": true,
        });
        
        if let Some(threshold) = score_threshold {
            body["score_threshold"] = serde_json::json!(threshold);
        }
        
        let response = self.http_client
            .post(&url)
            .json(&body)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            anyhow::bail!("Qdrant search failed ({}): {}", status, error_text);
        }
        
        let search_response: QdrantSearchResponse = response.json().await?;
        
        let results = search_response
            .result
            .into_iter()
            .map(|point| self.parse_scored_point(point))
            .collect::<Result<Vec<_>>>()?;

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

    async fn search_by_text(
        &self,
        query_text: &str,
        limit: u64,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        let query_vector = generate_embedding(query_text).await?;
        self.search_with_score(query_vector, limit, score_threshold)
            .await
    }
}

struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    fn new() -> Result<Self> {
        println!("📦 Loading embedding model (first time may take a while)...");
        
        let device = Device::Cpu;
        
        let repo = Repo::with_revision(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".to_string(),
            RepoType::Model,
            "main".to_string(),
        );
        
        let api = Api::new()?;
        let repo = api.repo(repo);
        
        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;
        
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = BertModel::load(vb, &config)?;
        
        println!("   ✅ Model loaded successfully");
        
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }
    
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        let token_ids = Tensor::new(tokens.get_ids(), &self.device)?
            .unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
        
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = normalize_l2(&embeddings)?;
        
        let embedding_vec = embeddings.squeeze(0)?.to_vec1::<f32>()?;
        
        Ok(embedding_vec)
    }
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

static EMBEDDING_MODEL: OnceLock<Arc<EmbeddingModel>> = OnceLock::new();

async fn generate_embedding(text: &str) -> Result<Vec<f32>> {
    println!("🔮 Generating embedding with Candle (Pure Rust)...");
    
    let model = EMBEDDING_MODEL.get_or_init(|| {
        Arc::new(EmbeddingModel::new().expect("Failed to load embedding model"))
    }).clone();
    
    let text_owned = text.to_string();
    let embedding = tokio::task::spawn_blocking(move || {
        model.encode(&text_owned)
    }).await??;
    
    println!("   ✅ Generated embedding (dim: {})", embedding.len());
    
    Ok(embedding)
}

#[tokio::main]
async fn main() -> Result<()> {
    let qdrant_host = env::var("QDRANT_HOST").unwrap_or_else(|_| "localhost".to_string());
    let qdrant_port: u16 = env::var("QDRANT_PORT")
        .unwrap_or_else(|_| "6333".to_string())
        .parse()?;
    let collection_name = env::var("COLLECTION_NAME")
        .unwrap_or_else(|_| "banking_statement".to_string());

    println!("🔍 Vector Search with Score - Qdrant Client");
    println!("   Host: {}:{}", qdrant_host, qdrant_port);
    println!("   Collection: {}", collection_name);
    println!();

    let client = VectorSearchClient::new(&qdrant_host, qdrant_port, &collection_name).await?;

    let query_text = "balance";
    println!("🔎 Searching for: '{}'", query_text);
    println!();

    let results = client
        .search_by_text(query_text, 10, None)
        .await?;

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
