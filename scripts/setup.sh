#!/bin/bash

# Setup script for Travel Agency RAG System

echo "🚀 Setting up Travel Agency RAG System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Generate sample data
echo "📊 Generating sample data..."
cd data
python generate_sample_data.py
cd ..

# Build indexes
echo "🔨 Building search indexes..."
echo "This may take a few minutes due to LLM API calls..."
cd indexing
python index_builder.py
cd ..

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and add your OpenAI API key"
echo "2. Run: python -m app.main"
echo "3. Open http://localhost:8000 in your browser"

