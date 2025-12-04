#!/bin/bash

# Run the FastAPI server

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the server
python -m app.main

