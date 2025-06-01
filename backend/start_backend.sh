#!/bin/bash

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start the Flask backend server
echo "Starting Flask backend server..."
python app.py
