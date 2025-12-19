# OrionRs LLM Setup Guide

Setup สำหรับ Ollama (SeaLLM/Typhoon) และ Qdrant Vector Database

## Prerequisites

- Docker และ Docker Compose
- RAM อย่างน้อย 8GB (16GB แนะนำ)
- Storage อย่างน้อย 20GB สำหรับ models
- macOS, Linux, หรือ Windows with WSL2

## Configuration

1. **Copy .env.example เป็น .env**
```bash
cp .env.example .env
```

2. **แก้ไข .env ตามต้องการ**
```bash
# Models ที่จะติดตั้ง (comma-separated)
OLLAMA_MODELS=seallm:7b,typhoon:13b

# Ollama API URL
OLLAMA_BASE_URL=http://localhost:11434

# Qdrant Vector Database URL
QDRANT_URL=http://localhost:6333
```

## การใช้งาน

### 1. Start Services
```bash
docker-compose up -d
```

### 2. ดาวน์โหลด Models
```bash
# เข้าไปใน Ollama container
docker exec -it orion-ollama ollama pull seallm:7b

# หรือ Typhoon (ใหญ่กว่า)
docker exec -it orion-ollama ollama pull typhoon:13b

# ดู models ที่ติดตั้งแล้ว
docker exec -it orion-ollama ollama list
```

### 3. ตรวจสอบ Logs
```bash
# ดู logs ทั้งหมด
docker-compose logs -f

# ดู logs เฉพาะ Ollama
docker-compose logs -f ollama

# ดู logs เฉพาะ Qdrant
docker-compose logs -f qdrant
```

### 4. ตรวจสอบสถานะ
```bash
docker-compose ps
```

### Stop Services
```bash
docker-compose down
```

### Stop และลบ volumes (ลบ models ทั้งหมด)
```bash
docker-compose down -v
```

## API Endpoints

### Ollama API
- **Base URL**: `http://localhost:11434`
- **Generate**: `http://localhost:11434/api/generate`
- **Chat**: `http://localhost:11434/api/chat`
- **List Models**: `http://localhost:11434/api/tags`

### Ollama Web UI
- **Web Interface**: `http://localhost:3000`
- ใช้งานผ่าน browser ได้เลย (ChatGPT-like interface)

### Qdrant Vector Database
- **REST API**: `http://localhost:6333`
- **Dashboard**: `http://localhost:6333/dashboard`
- **gRPC API**: `localhost:6334`

## ตัวอย่างการใช้งาน

### 1. ทดสอบ Ollama
```bash
# Generate text
curl http://localhost:11434/api/generate -d '{
  "model": "seallm:7b",
  "prompt": "สวัสดีครับ คุณชื่ออะไร",
  "stream": false
}'

# Chat (แนะนำ)
curl http://localhost:11434/api/chat -d '{
  "model": "seallm:7b",
  "messages": [
    {"role": "user", "content": "วิเคราะห์ข้อมูล JSON ให้หน่อย"}
  ],
  "stream": false
}'
```

### 2. ทดสอบ Qdrant
```bash
# สร้าง collection
curl -X PUT http://localhost:6333/collections/test \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }'

# ดู collections
curl http://localhost:6333/collections
```

### 3. Python Example
```python
import requests
import json

# Ollama Chat
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "seallm:7b",
        "messages": [
            {"role": "user", "content": "วิเคราะห์ข้อมูล JSON นี้ให้หน่อย"}
        ],
        "stream": False
    }
)
print(response.json()["message"]["content"])

# หรือใช้ ollama-python library
from ollama import Client

client = Client(host='http://localhost:11434')
response = client.chat(model='seallm:7b', messages=[
    {'role': 'user', 'content': 'สวัสดีครับ'}
])
print(response['message']['content'])

# Qdrant
from qdrant_client import QdrantClient

qdrant = QdrantClient(host="localhost", port=6333)
collections = qdrant.get_collections()
print(collections)
```

## Resource Requirements

### SeaLLM 7B
- **Storage**: ~4GB (model)
- **RAM**: 8GB+ (CPU mode)
- **Recommended**: 16GB RAM

### Typhoon 13B
- **Storage**: ~7GB (model)
- **RAM**: 16GB+
- **Recommended**: 32GB RAM

### Qdrant
- **Storage**: ขึ้นอยู่กับข้อมูล
- **RAM**: 2-4GB (base)

### Total System Requirements
- **Minimum**: 8GB RAM, 20GB storage
- **Recommended**: 16GB+ RAM, 50GB storage

## Troubleshooting

### Ollama ไม่ start
```bash
# ดู logs
docker-compose logs ollama

# ตรวจสอบว่า container ทำงานหรือไม่
docker ps | grep ollama

# Restart service
docker-compose restart ollama
```

### Model ดาวน์โหลดไม่สำเร็จ
```bash
# ลองดาวน์โหลดใหม่
docker exec -it orion-ollama ollama pull seallm:7b

# ตรวจสอบ internet connection
curl -I https://ollama.ai

# ดู available models
docker exec -it orion-ollama ollama list
```

### Out of Memory
- ใช้ model เล็กกว่า (7B แทน 13B)
- ปิด applications อื่นๆ
- เพิ่ม RAM ให้ Docker (Docker Desktop > Settings > Resources)

### Response ช้า
- ปกติสำหรับ CPU mode
- ลด context length ในการ query
- ใช้ model เล็กกว่า
- พิจารณาใช้ Apple Silicon Mac (M1/M2/M3) จะเร็วกว่า Intel Mac มาก

## Available Models

### SeaLLM Models
```bash
# SeaLLM 7B (แนะนำ)
docker exec -it orion-ollama ollama pull seallm:7b

# SeaLLM 13B (ต้องการ RAM มากกว่า)
docker exec -it orion-ollama ollama pull seallm:13b
```

### Typhoon Models (Thai-specific)
```bash
# Typhoon 7B
docker exec -it orion-ollama ollama pull typhoon:7b

# Typhoon 13B
docker exec -it orion-ollama ollama pull typhoon:13b
```

### Other Thai-capable Models
```bash
# Llama 3.1 (รองรับภาษาไทย)
docker exec -it orion-ollama ollama pull llama3.1:8b

# Qwen 2.5 (รองรับภาษาไทยได้ดี)
docker exec -it orion-ollama ollama pull qwen2.5:7b
```

## Next Steps

1. ทดสอบ Ollama API และ Web UI
2. สร้าง RAG pipeline ด้วย Qdrant
3. Load ข้อมูล JSON (statement.json) เข้า vector database
4. สร้าง application ที่ใช้ LLM วิเคราะห์ข้อมูล banking/fintech
5. Fine-tune model สำหรับ domain-specific tasks

## Quick Start Commands

```bash
# 1. Start all services
docker-compose up -d

# 2. Download SeaLLM model
docker exec -it orion-ollama ollama pull seallm:7b

# 3. Test via Web UI
open http://localhost:3000

# 4. Test via API
curl http://localhost:11434/api/chat -d '{
  "model": "seallm:7b",
  "messages": [{"role": "user", "content": "สวัสดีครับ"}],
  "stream": false
}'

# 5. Access Qdrant Dashboard
open http://localhost:6333/dashboard
```

## References

- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Open WebUI](https://github.com/open-webui/open-webui)
- [SeaLLM Models](https://huggingface.co/SeaLLMs)
- [Typhoon Models](https://huggingface.co/scb10x)
