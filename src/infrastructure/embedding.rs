use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

pub struct EmbeddingService {
    model: Arc<EmbeddingModel>,
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

impl EmbeddingService {
    pub fn new() -> Self {
        let model = EMBEDDING_MODEL.get_or_init(|| {
            Arc::new(EmbeddingModel::new().expect("Failed to load embedding model"))
        }).clone();
        
        Self { model }
    }
    
    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        println!("🔮 Generating embedding with Candle (Pure Rust)...");
        
        let model = self.model.clone();
        let text_owned = text.to_string();
        
        let embedding = tokio::task::spawn_blocking(move || {
            model.encode(&text_owned)
        }).await??;
        
        println!("   ✅ Generated embedding (dim: {})", embedding.len());
        
        Ok(embedding)
    }
}

impl Default for EmbeddingService {
    fn default() -> Self {
        Self::new()
    }
}
