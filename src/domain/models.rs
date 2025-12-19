use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub text: String,
    pub source_file: String,
    pub collection: String,
}

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub text: String,
    pub limit: u64,
    pub score_threshold: Option<f32>,
}

impl SearchQuery {
    pub fn new(text: impl Into<String>, limit: u64) -> Self {
        Self {
            text: text.into(),
            limit,
            score_threshold: None,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }
}
