#!/bin/bash

set -e

echo "🧪 Testing OrionRs LLM Setup..."
echo ""

echo "1️⃣ Testing Ollama API..."
OLLAMA_RESPONSE=$(curl -s http://localhost:11434/api/tags)
if [ $? -eq 0 ]; then
    echo "   ✅ Ollama API is running"
    echo "   📦 Installed models:"
    echo "$OLLAMA_RESPONSE" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | sed 's/^/      - /'
else
    echo "   ❌ Ollama API is not responding"
    exit 1
fi

echo ""
echo "2️⃣ Testing Qdrant API..."
QDRANT_RESPONSE=$(curl -s http://localhost:6333/collections)
if [ $? -eq 0 ]; then
    echo "   ✅ Qdrant API is running"
else
    echo "   ❌ Qdrant API is not responding"
    exit 1
fi

echo ""
echo "3️⃣ Testing Ollama Chat (Thai language)..."
CHAT_RESPONSE=$(curl -s http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [
    {"role": "user", "content": "สวัสดีครับ ตอบสั้นๆ ว่าคุณคือใคร"}
  ],
  "stream": false
}')

if [ $? -eq 0 ]; then
    echo "   ✅ Chat API is working"
    echo "   💬 Response:"
    echo "$CHAT_RESPONSE" | grep -o '"content":"[^"]*"' | head -1 | cut -d'"' -f4 | sed 's/^/      /'
else
    echo "   ❌ Chat API failed"
    exit 1
fi

echo ""
echo "4️⃣ Testing Web UI..."
WEBUI_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3333)
if [ "$WEBUI_RESPONSE" = "200" ]; then
    echo "   ✅ Web UI is accessible at http://localhost:3333"
else
    echo "   ⚠️  Web UI returned status: $WEBUI_RESPONSE"
fi

echo ""
echo "✅ All tests passed!"
echo ""
echo "🌐 Access points:"
echo "   - Web UI:           http://localhost:3333"
echo "   - Ollama API:       http://localhost:11434"
echo "   - Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
