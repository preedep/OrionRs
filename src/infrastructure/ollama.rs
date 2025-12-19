use anyhow::Result;
use async_trait::async_trait;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};

use crate::domain::{ChatRequest, ChatResponse, LLMService};

pub struct OllamaClient {
    http_client: reqwest::Client,
    base_url: String,
}

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
    model: String,
}

#[derive(Deserialize)]
struct OllamaResponseMessage {
    content: String,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        info!("Initializing Ollama client");
        debug!("Ollama URL: {}", base_url);
        
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?;

        info!("Ollama client initialized successfully");

        Ok(Self {
            http_client,
            base_url,
        })
    }
}

#[async_trait]
impl LLMService for OllamaClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        info!("Sending chat request to Ollama");
        debug!("Model: {}", request.model);
        debug!("Messages count: {}", request.messages.len());
        debug!("Temperature: {:?}", request.temperature);
        
        let url = format!("{}/api/chat", self.base_url);

        let messages: Vec<OllamaMessage> = request
            .messages
            .into_iter()
            .map(|msg| OllamaMessage {
                role: msg.role,
                content: msg.content,
            })
            .collect();

        let options = if request.temperature.is_some() || request.max_tokens.is_some() {
            Some(OllamaOptions {
                temperature: request.temperature,
                num_predict: request.max_tokens,
            })
        } else {
            None
        };

        let ollama_request = OllamaChatRequest {
            model: request.model.clone(),
            messages,
            stream: false,
            options,
        };

        let start = std::time::Instant::now();
        let response = self
            .http_client
            .post(&url)
            .json(&ollama_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            warn!("Ollama request failed with status {}: {}", status, error_text);
            anyhow::bail!("Ollama request failed ({}): {}", status, error_text);
        }

        let ollama_response: OllamaChatResponse = response.json().await?;
        
        let duration = start.elapsed();
        info!("Ollama response received (took: {:.2}s)", duration.as_secs_f64());
        debug!("Response length: {} chars", ollama_response.message.content.len());

        Ok(ChatResponse {
            content: ollama_response.message.content,
            model: ollama_response.model,
        })
    }
}
