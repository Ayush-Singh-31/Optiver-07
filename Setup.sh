#!/usr/bin/env bash
# This script creates a Python virtual environment named 'volt', activates it, and installs dependencies from Requirments.txt.

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "python3 is not installed. Please install Python 3."
    exit 1
fi

# Create a virtual environment named 'volt' if it doesn't exist.
if [ ! -d "volt" ]; then
    echo "Creating virtual environment 'volt'..."
    python3 -m venv volt
fi

# Activate the virtual environment.
# For Windows (using Git Bash or similar), the activation script is in volt/Scripts.
# For macOS/Linux, it's in volt/bin.
UNAME_OUT="$(uname -s)"
case "${UNAME_OUT}" in
    Linux*|Darwin*)
        source volt/bin/activate
        ;;
    MINGW*|CYGWIN*|MSYS*)
        source volt/Scripts/activate
        ;;
    *)
        echo "Unknown OS. Please activate your virtual environment manually."
        exit 1
        ;;
esac

# Upgrade pip to the latest version.
pip install --upgrade pip

# Install dependencies from Requirments.txt if it exists.
if [ -f "Requirments.txt" ]; then
    echo "Installing dependencies from Requirments.txt..."
    pip install -r Requirments.txt
else
    echo "Requirments.txt not found."
fi

echo "Setup complete. Virtual environment 'volt' is activated."