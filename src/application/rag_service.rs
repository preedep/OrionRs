use anyhow::Result;

use crate::domain::{ChatMessage, ChatRequest, ChatResponse, LLMService, SearchQuery, SearchResult};
use crate::infrastructure::{EmbeddingService, QdrantClient};

pub struct RagService<L: LLMService> {
    embedding_service: EmbeddingService,
    vector_db: QdrantClient,
    llm_service: L,
}

impl<L: LLMService> RagService<L> {
    pub fn new(
        embedding_service: EmbeddingService,
        vector_db: QdrantClient,
        llm_service: L,
    ) -> Self {
        Self {
            embedding_service,
            vector_db,
            llm_service,
        }
    }

    async fn search_documents(&self, query: &str, limit: u64) -> Result<Vec<SearchResult>> {
        let search_query = SearchQuery::new(query, limit);
        
        let query_vector = self.embedding_service
            .generate_embedding(&search_query.text)
            .await?;
        
        let results = self.vector_db
            .search(query_vector, search_query.limit, search_query.score_threshold)
            .await?;
        
        Ok(results)
    }

    fn build_context(&self, results: &[SearchResult]) -> String {
        let mut context = String::new();
        
        for (idx, result) in results.iter().enumerate() {
            context.push_str(&format!(
                "\n[Document {}] (Score: {:.4}, Source: {})\n{}\n",
                idx + 1,
                result.score,
                result.source_file,
                result.text
            ));
        }
        
        context
    }

    pub async fn ask(
        &self,
        question: &str,
        model: &str,
        num_results: u64,
    ) -> Result<RagResponse> {
        println!("🔍 Searching for relevant documents...");
        let search_results = self.search_documents(question, num_results).await?;
        
        println!("   ✅ Found {} relevant documents", search_results.len());
        
        let context = self.build_context(&search_results);
        
        let system_prompt = format!(
            "You are a helpful assistant that answers questions based on the provided context. \
            Use the following documents to answer the user's question. \
            If the answer cannot be found in the context, say so.\n\
            \nContext:{}\n",
            context
        );

        let chat_request = ChatRequest::new(model)
            .add_message(ChatMessage::system(system_prompt))
            .add_message(ChatMessage::user(question))
            .with_temperature(0.7);

        println!("🤖 Generating answer with {}...", model);
        let response = self.llm_service.chat(chat_request).await?;
        
        Ok(RagResponse {
            answer: response.content,
            sources: search_results,
            model: response.model,
        })
    }
}

#[derive(Debug)]
pub struct RagResponse {
    pub answer: String,
    pub sources: Vec<SearchResult>,
    pub model: String,
}
