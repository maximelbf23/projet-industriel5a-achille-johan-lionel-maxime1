#!/bin/bash
# Script to run the Streamlit app using the local virtual environment

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Check if .venv exists
if [ -d ".venv" ]; then
    echo "Starting Streamlit app using .venv..."
    ./.venv/bin/python -m streamlit run "Profil de température Aube.py"
else
    echo "Virtual environment (.venv) not found!"
    echo "Please ensure you have set up the environment."
    exit 1
fi
