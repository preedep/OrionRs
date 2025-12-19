use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub qdrant_host: String,
    pub qdrant_port: u16,
    pub collection_name: String,
    pub ollama_base_url: String,
    pub ollama_model: String,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let qdrant_host = env::var("QDRANT_HOST")
            .unwrap_or_else(|_| "localhost".to_string());
        
        let qdrant_port: u16 = env::var("QDRANT_PORT")
            .unwrap_or_else(|_| "6333".to_string())
            .parse()?;
        
        let collection_name = env::var("COLLECTION_NAME")
            .unwrap_or_else(|_| "banking_scb_st_stpost".to_string());

        let ollama_base_url = env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());

        let ollama_model = env::var("OLLAMA_MODEL")
            .unwrap_or_else(|_| "llama3.2".to_string());

        Ok(Self {
            qdrant_host,
            qdrant_port,
            collection_name,
            ollama_base_url,
            ollama_model,
        })
    }
}
