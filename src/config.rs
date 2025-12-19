use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub qdrant_host: String,
    pub qdrant_port: u16,
    pub collection_name: String,
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

        Ok(Self {
            qdrant_host,
            qdrant_port,
            collection_name,
        })
    }
}
