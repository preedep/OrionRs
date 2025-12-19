#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "banking")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DATA_DIR = os.getenv("DATA_DIR", "datasource/raw_json")

print(f"🚀 Starting RAG import process...")
print(f"   Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
print(f"   Collection prefix: {COLLECTION_PREFIX}")
print(f"   Model: {EMBEDDING_MODEL}")
print(f"   Data directory: {DATA_DIR}")
print(f"   Strategy: One collection per JSON file")
print()

print("📦 Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"   ✅ Model loaded (dimension: {embedding_dim})")
print()

print("🔌 Connecting to Qdrant...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print("   ✅ Connected to Qdrant")
print()

def process_json_item(item: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    text_parts = []
    
    if isinstance(item, dict):
        for key, value in item.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, (int, float, bool)):
                text_parts.append(f"{key}: {value}")
    
    text = " | ".join(text_parts)
    
    return {
        "text": text,
        "source_file": source_file,
        "original_data": item
    }

def chunk_text(text: str, max_length: int = 500) -> List[str]:
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

print("📂 Loading JSON files...")
data_path = Path(DATA_DIR)
json_files = list(data_path.glob("*.json"))

if not json_files:
    print(f"❌ No JSON files found in {DATA_DIR}")
    sys.exit(1)

print(f"   Found {len(json_files)} JSON files")
print()

total_vectors = 0
collections_created = []

for json_file in json_files:
    file_base_name = json_file.stem
    collection_name = f"{COLLECTION_PREFIX}_{file_base_name}"
    
    print(f"📄 Processing {json_file.name} → Collection: {collection_name}")
    
    existing_collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in existing_collections)
    
    if collection_exists:
        print(f"   ⚠️  Collection '{collection_name}' already exists")
        response = input(f"   Recreate? (y/N): ").strip().lower()
        if response == 'y':
            print(f"   🗑️  Deleting existing collection...")
            client.delete_collection(collection_name=collection_name)
            collection_exists = False
        else:
            print(f"   ℹ️  Skipping {json_file.name}")
            continue
    
    if not collection_exists:
        print(f"   📝 Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            print(f"   ⚠️  Skipping {json_file.name}: unexpected format")
            continue
        
        print(f"   Found {len(items)} items")
        
        file_points = []
        point_id = 0
        
        for idx, item in enumerate(items):
            processed = process_json_item(item, json_file.name)
            text = processed["text"]
            
            if not text or len(text.strip()) < 10:
                continue
            
            chunks = chunk_text(text, max_length=500)
            
            for chunk_idx, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source_file": json_file.name,
                        "collection": collection_name,
                        "item_index": idx,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "original_data": processed["original_data"]
                    }
                )
                
                file_points.append(point)
                point_id += 1
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(items)} items...")
        
        print(f"   💾 Uploading {len(file_points)} vectors to '{collection_name}'...")
        
        batch_size = 100
        for i in range(0, len(file_points), batch_size):
            batch = file_points[i:i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            if len(file_points) > batch_size:
                print(f"      Uploaded {min(i + batch_size, len(file_points))}/{len(file_points)} vectors...")
        
        total_vectors += len(file_points)
        collections_created.append({
            'name': collection_name,
            'file': json_file.name,
            'vectors': len(file_points)
        })
        
        print(f"   ✅ Completed {json_file.name} ({len(file_points)} vectors)")
        print()
    
    except Exception as e:
        print(f"   ❌ Error processing {json_file.name}: {e}")
        continue

print()
print("✅ Import completed successfully!")
print()
print(f"📊 Summary:")
print(f"   - Total collections: {len(collections_created)}")
print(f"   - Total vectors: {total_vectors}")
print(f"   - Embedding dimension: {embedding_dim}")
print()
print("📚 Collections created:")
for coll in collections_created:
    print(f"   - {coll['name']}: {coll['vectors']} vectors (from {coll['file']})")
print()
print(f"🔍 Test query (all collections):")
print(f"   python scripts/query_qdrant.py \"ค้นหาข้อมูลเกี่ยวกับบัญชี\"")
print()
print(f"🔍 Test query (specific collection):")
print(f"   python scripts/query_qdrant.py \"ค้นหาข้อมูล\" --collection {collections_created[0]['name'] if collections_created else 'banking_statement'}")
print()
