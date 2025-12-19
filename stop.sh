#!/bin/bash

set -e

echo "🛑 Stopping OrionRs LLM services..."
echo ""

docker-compose down

echo ""
echo "✅ All services stopped"
echo ""
echo "💡 To remove all data (including downloaded models), run:"
echo "   docker-compose down -v"
echo ""
