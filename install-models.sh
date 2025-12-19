#!/bin/bash

set -e

echo "📥 Installing LLM Models..."
echo ""

echo "Available models:"
echo "  1) Llama 3.2 3B (Recommended, fast, ~2GB)"
echo "  2) Llama 3.1 8B (Better quality, ~4.7GB)"
echo "  3) Qwen 2.5 7B (Great for Thai, ~4.7GB)"
echo "  4) Qwen 2.5 14B (Best quality, ~9GB)"
echo "  5) Gemma 2 9B (Google model, ~5.5GB)"
echo "  6) Phi 3.5 (Microsoft, ~2.2GB)"
echo "  7) All recommended (Llama 3.2 3B + Qwen 2.5 7B)"
echo ""

read -p "Select model to install (1-7): " choice

case $choice in
    1)
        echo "📥 Installing Llama 3.2 3B..."
        docker exec -it orion-ollama ollama pull llama3.2:3b
        ;;
    2)
        echo "📥 Installing Llama 3.1 8B..."
        docker exec -it orion-ollama ollama pull llama3.1:8b
        ;;
    3)
        echo "📥 Installing Qwen 2.5 7B..."
        docker exec -it orion-ollama ollama pull qwen2.5:7b
        ;;
    4)
        echo "📥 Installing Qwen 2.5 14B..."
        docker exec -it orion-ollama ollama pull qwen2.5:14b
        ;;
    5)
        echo "📥 Installing Gemma 2 9B..."
        docker exec -it orion-ollama ollama pull gemma2:9b
        ;;
    6)
        echo "📥 Installing Phi 3.5..."
        docker exec -it orion-ollama ollama pull phi3.5
        ;;
    7)
        echo "📥 Installing Llama 3.2 3B..."
        docker exec -it orion-ollama ollama pull llama3.2:3b
        echo ""
        echo "📥 Installing Qwen 2.5 7B..."
        docker exec -it orion-ollama ollama pull qwen2.5:7b
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ Installation complete!"
echo ""
echo "📦 Installed models:"
docker exec -it orion-ollama ollama list
echo ""
