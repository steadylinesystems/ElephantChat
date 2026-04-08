#!/usr/bin/env bash
# LMChat launcher
cd "$(dirname "$0")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LMChat — Local AI Interface"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ python3 not found. Please install Python 3.10+."
  exit 1
fi

# Install deps if needed
echo "📦 Checking dependencies..."
pip3 install fastapi uvicorn python-multipart pypdf httpx aiofiles rank-bm25 -q

echo "✅ Dependencies ready"
echo ""
echo "🚀 Starting LMChat at http://localhost:8099"
echo "   Make sure LM Studio is running with a model loaded."
echo "   Press Ctrl+C to stop."
echo ""

python3 -m uvicorn backend:app --host 0.0.0.0 --port 8099 --reload
