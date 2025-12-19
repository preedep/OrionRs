#!/bin/bash

set -e

echo "🚀 Starting OrionRs LLM Setup..."
echo ""

if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
fi

echo "🐳 Starting Docker containers..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

echo ""
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "📥 Downloading Llama 3.2 3B model (Thai-capable)..."
echo "   (This may take a few minutes on first run)"
docker exec -it orion-ollama ollama pull llama3.2:3b

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Available services:"
echo "   - Ollama API:      http://localhost:11434"
echo "   - Web UI:          http://localhost:3333"
echo "   - Qdrant API:      http://localhost:6333"
echo "   - Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "🧪 To test the setup, run: ./test.sh"
echo "🛑 To stop services, run: ./stop.sh"
echo ""
echo "💡 Quick test command:"
echo '   curl http://localhost:11434/api/chat -d '"'"'{"model":"llama3.2:3b","messages":[{"role":"user","content":"สวัสดีครับ"}],"stream":false}'"'"''
echo ""
