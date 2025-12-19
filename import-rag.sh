#!/bin/bash

set -e

echo "🚀 RAG Import Script for OrionRs"
echo ""

if [ ! -d "datasource/raw_json" ]; then
    echo "❌ Error: datasource/raw_json directory not found"
    exit 1
fi

JSON_COUNT=$(find datasource/raw_json -name "*.json" | wc -l | tr -d ' ')
if [ "$JSON_COUNT" -eq 0 ]; then
    echo "❌ Error: No JSON files found in datasource/raw_json"
    exit 1
fi

echo "📂 Found $JSON_COUNT JSON files in datasource/raw_json"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed"
    exit 1
fi

echo "🔍 Checking Python environment..."
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "📥 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r scripts/requirements.txt

echo ""
echo "🔍 Checking if Qdrant is running..."
if ! curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "❌ Error: Qdrant is not running"
    echo "   Please start services first: ./start.sh"
    exit 1
fi
echo "   ✅ Qdrant is running"

echo ""
echo "📊 Starting import process..."
echo ""

python3 scripts/import_to_qdrant.py

echo ""
echo "✅ Import completed!"
echo ""
echo "🧪 Test the RAG system:"
echo "   source venv/bin/activate"
echo "   python scripts/query_qdrant.py \"ค้นหาข้อมูลเกี่ยวกับบัญชี\""
echo ""
