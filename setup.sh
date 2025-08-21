#!/bin/bash

# This script automates the setup process for the AI Market Analyst RAG System.
# It creates a virtual environment, installs dependencies, and provides
# final instructions for the user.

# --- Style Definitions ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}--- Starting AI Market Analyst Setup ---${NC}"

# --- Step 1: Check for Python ---
echo "Step 1: Checking for Python 3.9+..."
if ! command -v python3 &> /dev/null || ! python3 -c 'import sys; assert sys.version_info >= (3, 9)' &> /dev/null; then
    echo -e "${YELLOW}Error: Python 3.9 or higher is not installed or not in the PATH.${NC}"
    echo "Please install Python 3.9+ and try again."
    exit 1
fi
echo -e "${GREEN}Python check passed!${NC}"
echo ""

# --- Step 2: Create and Activate Virtual Environment ---
echo "Step 2: Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment in './venv'..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"
echo ""

# --- Step 3: Install Dependencies ---
echo "Step 3: Installing required Python packages..."
pip install --upgrade pip
if pip install -r requirements.txt; then
    echo -e "${GREEN}Core dependencies installed successfully.${NC}"
else
    echo -e "${YELLOW}Error: Failed to install core dependencies from requirements.txt.${NC}"
    exit 1
fi

# Explicitly handle the NumPy version to prevent conflicts
echo "Ensuring compatible NumPy version is installed..."
pip install "numpy<2.0"
echo -e "${GREEN}Dependencies installed.${NC}"
echo ""

# --- Step 4: LLM Model Setup (Manual Step) ---
MODEL_NAME="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
echo -e "Step 4: ${YELLOW}Manual Action Required - Download the LLM${NC}"
if [ -f "$MODEL_NAME" ]; then
    echo -e "${GREEN}Model file '$MODEL_NAME' already found in the project directory.${NC}"
else
    echo "The application requires the Llama 3 GGUF model file."
    echo "Please download '${MODEL_NAME}'"
    echo "You can typically find it on Hugging Face."
    echo ""
    echo -e "➡️  ${YELLOW}Once downloaded, place the file in this directory before running the app.${NC}"
fi
echo ""

# --- Final Instructions ---
echo -e "${GREEN}--- Setup Complete! ---${NC}"
echo ""
echo "You are now ready to run the application."
echo "Use the following command to start the server:"
echo -e "${BLUE}uvicorn app:app --host 0.0.0.0 --port 8000${NC}"
echo ""
echo "Then, open your web browser and go to: ${GREEN}http://localhost:8000${NC}"

