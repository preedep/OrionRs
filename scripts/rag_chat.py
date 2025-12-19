#!/usr/bin/env python3

import os
import sys
import argparse
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "banking")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

def search_documents(query: str, top_k: int = 3, specific_collection: str = None):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    all_collections = client.get_collections().collections
    banking_collections = [c.name for c in all_collections if c.name.startswith(COLLECTION_PREFIX)]
    
    if not banking_collections:
        print(f"❌ No collections found with prefix '{COLLECTION_PREFIX}'")
        return []
    
    if specific_collection:
        if specific_collection in banking_collections:
            collections_to_search = [specific_collection]
        else:
            print(f"⚠️  Collection '{specific_collection}' not found, searching all collections")
            collections_to_search = banking_collections
    else:
        collections_to_search = banking_collections
    
    query_vector = model.encode(query).tolist()
    
    all_results = []
    for collection_name in collections_to_search:
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            for result in results:
                result.payload['_collection'] = collection_name
                all_results.append(result)
        except Exception as e:
            print(f"⚠️  Error searching {collection_name}: {e}")
    
    all_results.sort(key=lambda x: x.score, reverse=True)
    return all_results[:top_k]

def chat_with_context(query: str, context_docs: list):
    context_text = "\n\n".join([
        f"Document {idx+1} (from {doc.payload.get('source_file', 'unknown')}):\n{doc.payload.get('text', '')}"
        for idx, doc in enumerate(context_docs)
    ])
    
    prompt = f"""คุณเป็น AI Assistant ที่เชี่ยวชาญด้านข้อมูลธนาคารและการเงิน

ข้อมูลที่เกี่ยวข้อง:
{context_text}

คำถาม: {query}

กรุณาตอบคำถามโดยอ้างอิงจากข้อมูลที่ให้มา หากข้อมูลไม่เพียงพอให้บอกว่าไม่มีข้อมูล"""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
    )
    
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return f"Error: {response.status_code}"

def main():
    parser = argparse.ArgumentParser(description='RAG Chat with Ollama and Qdrant')
    parser.add_argument('question', nargs='+', help='Your question')
    parser.add_argument('--collection', '-c', help='Specific collection to search')
    parser.add_argument('--top-k', '-k', type=int, default=3, help='Number of documents to retrieve (default: 3)')
    
    args = parser.parse_args()
    question = " ".join(args.question)
    
    print(f"🔍 Question: {question}")
    print()
    
    print("📚 Searching relevant documents...")
    docs = search_documents(question, top_k=args.top_k, specific_collection=args.collection)
    
    if not docs:
        print("❌ No documents found. Please run import first: ./import-rag.sh")
        sys.exit(1)
    
    print(f"   Found {len(docs)} relevant documents")
    print()
    
    print("🤖 Generating answer with LLM...")
    print()
    
    answer = chat_with_context(question, docs)
    
    print("=" * 80)
    print("💬 Answer:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    print()
    
    print("📄 Sources:")
    for idx, doc in enumerate(docs, 1):
        collection = doc.payload.get('_collection', 'unknown')
        source_file = doc.payload.get('source_file', 'unknown')
        print(f"   {idx}. [{collection}] {source_file} (score: {doc.score:.4f})")
    print()

if __name__ == "__main__":
    main()
