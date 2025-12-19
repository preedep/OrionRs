#!/usr/bin/env python3

import os
import sys
import argparse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "banking")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

parser = argparse.ArgumentParser(description='Query Qdrant vector database')
parser.add_argument('query', nargs='+', help='Search query')
parser.add_argument('--collection', '-c', help='Specific collection to search (default: search all)')
parser.add_argument('--limit', '-l', type=int, default=5, help='Number of results per collection (default: 5)')

args = parser.parse_args()
query = " ".join(args.query)

print(f"🔍 Searching for: {query}")
print()

print("📦 Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("🔌 Connecting to Qdrant...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

all_collections = client.get_collections().collections
banking_collections = [c.name for c in all_collections if c.name.startswith(COLLECTION_PREFIX)]

if not banking_collections:
    print(f"❌ No collections found with prefix '{COLLECTION_PREFIX}'")
    print("   Run import first: ./import-rag.sh")
    sys.exit(1)

if args.collection:
    if args.collection in banking_collections:
        collections_to_search = [args.collection]
        print(f"📚 Searching in collection: {args.collection}")
    else:
        print(f"❌ Collection '{args.collection}' not found")
        print(f"   Available collections: {', '.join(banking_collections)}")
        sys.exit(1)
else:
    collections_to_search = banking_collections
    print(f"📚 Searching in {len(collections_to_search)} collections:")
    for coll in collections_to_search:
        print(f"   - {coll}")

print()
print("🔎 Generating query embedding...")
query_vector = model.encode(query).tolist()

all_results = []

for collection_name in collections_to_search:
    print(f"📊 Searching in '{collection_name}'...")
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=args.limit
        )
        for result in results:
            result.payload['_collection'] = collection_name
            all_results.append(result)
    except Exception as e:
        print(f"   ⚠️  Error searching {collection_name}: {e}")

all_results.sort(key=lambda x: x.score, reverse=True)
top_results = all_results[:args.limit * 2] if len(collections_to_search) > 1 else all_results

print()
print(f"✅ Found {len(top_results)} top results:")
print("=" * 80)

for idx, result in enumerate(top_results, 1):
    print(f"\n{idx}. Score: {result.score:.4f}")
    print(f"   Collection: {result.payload.get('_collection', 'N/A')}")
    print(f"   Source: {result.payload.get('source_file', 'N/A')}")
    print(f"   Text: {result.payload.get('text', 'N/A')[:200]}...")
    print("-" * 80)

print()
