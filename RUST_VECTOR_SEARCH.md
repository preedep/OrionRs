# Rust Vector Search with Qdrant

Rust implementation สำหรับ vector search พร้อม score ใน Qdrant

## Dependencies

```toml
[dependencies]
qdrant-client = "1.12"
tokio = { version = "1.42", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
```

## Features

- ✅ Vector search พร้อม score
- ✅ Score threshold filtering
- ✅ Async/await support
- ✅ Type-safe payload parsing
- ✅ Environment variable configuration

## Usage

### 1. Build

```bash
cargo build --release
```

### 2. Run

```bash
# ใช้ค่า default
cargo run

# หรือกำหนด environment variables
QDRANT_HOST=localhost \
QDRANT_PORT=6333 \
COLLECTION_NAME=banking_statement \
cargo run
```

### 3. ตัวอย่าง Output

```
🔍 Vector Search with Score - Qdrant Client
   Host: localhost:6333
   Collection: banking_statement

🔎 Searching for: 'AccountStatement'

✅ Found 5 results:
================================================================================

1. Score: 0.8542
   ID: 0
   Collection: banking_statement
   Source: statement.json
   Text: data_type: Array | field_name: AccountStatement | m_o_c: M...
--------------------------------------------------------------------------------

2. Score: 0.7234
   ...
```

## Code Structure

### VectorSearchClient

```rust
struct VectorSearchClient {
    client: QdrantClient,
    collection_name: String,
}

impl VectorSearchClient {
    // สร้าง client ใหม่
    async fn new(host: &str, port: u16, collection_name: &str) -> Result<Self>
    
    // ค้นหาด้วย vector พร้อม score
    async fn search_with_score(
        &self,
        query_vector: Vec<f32>,
        limit: u64,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>>
    
    // ค้นหาด้วย text (ต้อง implement embedding)
    async fn search_by_text(
        &self,
        query_text: &str,
        limit: u64,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>>
}
```

### SearchResult

```rust
struct SearchResult {
    id: u64,
    score: f32,
    text: String,
    source_file: String,
    collection: String,
}
```

## Embedding Generation

⚠️ **สำคัญ**: ตอนนี้ `generate_embedding()` เป็น placeholder

### ตัวเลือกสำหรับ Production:

#### 1. **Call Python Embedding Service (แนะนำ)**

```rust
use reqwest;

async fn generate_embedding(text: &str) -> Result<Vec<f32>> {
    let client = reqwest::Client::new();
    let response = client
        .post("http://localhost:8000/embed")
        .json(&serde_json::json!({ "text": text }))
        .send()
        .await?;
    
    let embedding: Vec<f32> = response.json().await?;
    Ok(embedding)
}
```

สร้าง Python service:
```python
# embedding_service.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@app.post("/embed")
async def embed(data: dict):
    text = data["text"]
    embedding = model.encode(text).tolist()
    return embedding
```

รัน service:
```bash
pip install fastapi uvicorn sentence-transformers
uvicorn embedding_service:app --port 8000
```

#### 2. **ใช้ rust-bert (ช้ากว่า)**

```rust
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

async fn generate_embedding(text: &str) -> Result<Vec<f32>> {
    let model = SentenceEmbeddingsBuilder::remote(
        SentenceEmbeddingsModelType::AllMiniLmL12V2
    )
    .create_model()?;
    
    let embeddings = model.encode(&[text])?;
    Ok(embeddings[0].clone())
}
```

#### 3. **ใช้ External API (OpenAI, Cohere, etc.)**

```rust
async fn generate_embedding(text: &str) -> Result<Vec<f32>> {
    // Call OpenAI Embeddings API
    // หรือ Cohere Embed API
    // ...
}
```

## Advanced Usage

### Search Multiple Collections

```rust
async fn search_all_collections(
    host: &str,
    port: u16,
    query_text: &str,
) -> Result<Vec<SearchResult>> {
    let collections = vec!["banking_statement", "banking_scb_st_stpost", "banking_scb_im_impost"];
    let mut all_results = Vec::new();
    
    for collection in collections {
        let client = VectorSearchClient::new(host, port, collection).await?;
        let results = client.search_by_text(query_text, 5, Some(0.5)).await?;
        all_results.extend(results);
    }
    
    // Sort by score
    all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    
    Ok(all_results)
}
```

### Filter by Payload

```rust
use qdrant_client::qdrant::{Condition, Filter, FieldCondition, Match};

async fn search_with_filter(
    &self,
    query_vector: Vec<f32>,
    source_file: &str,
) -> Result<Vec<SearchResult>> {
    let filter = Filter {
        must: vec![Condition {
            condition_one_of: Some(
                qdrant_client::qdrant::condition::ConditionOneOf::Field(
                    FieldCondition {
                        key: "source_file".to_string(),
                        r#match: Some(Match {
                            match_value: Some(
                                qdrant_client::qdrant::r#match::MatchValue::Keyword(
                                    source_file.to_string()
                                )
                            ),
                        }),
                        ..Default::default()
                    }
                )
            ),
        }],
        ..Default::default()
    };
    
    let search_points = SearchPoints {
        collection_name: self.collection_name.clone(),
        vector: query_vector,
        filter: Some(filter),
        limit: 10,
        with_payload: Some(WithPayloadSelector {
            selector_options: Some(
                qdrant_client::qdrant::with_payload_selector::SelectorOptions::Enable(true),
            ),
        }),
        ..Default::default()
    };
    
    let search_result = self.client.search_points(&search_points).await?;
    // ... parse results
}
```

## Testing

```bash
# ต้องมี Qdrant running และมีข้อมูลแล้ว
docker-compose up -d qdrant

# Import ข้อมูล (ถ้ายังไม่ได้ทำ)
./import-rag.sh

# Run Rust program
cargo run
```

## Performance Tips

1. **Connection Pooling**: Reuse `QdrantClient` instance
2. **Batch Search**: Search multiple queries at once
3. **Caching**: Cache embeddings for repeated queries
4. **Score Threshold**: ใช้ `score_threshold` เพื่อลด results ที่ไม่เกี่ยวข้อง

## Troubleshooting

### Error: "Connection refused"
```bash
# ตรวจสอบว่า Qdrant running
docker-compose ps
curl http://localhost:6333/collections
```

### Error: "Collection not found"
```bash
# Import ข้อมูลก่อน
./import-rag.sh
```

### Error: "Invalid vector dimension"
```bash
# ตรวจสอบว่า embedding model ใช้ dimension เดียวกับที่ import
# paraphrase-multilingual-MiniLM-L12-v2 = 384 dimensions
```

## Next Steps

1. Implement production embedding generation
2. Add error handling และ retry logic
3. Add logging (tracing, log crates)
4. Add metrics และ monitoring
5. Create REST API wrapper (actix-web, axum)
6. Add authentication
7. Deploy as microservice
