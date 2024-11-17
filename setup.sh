#!/usr/bin/env bash

# Function to check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python could not be found. Installing Python..."
        install_python
    else
        local python_path=$(which python3)
        echo "Python is installed at $python_path."
        echo "Python version: $(python3 --version)"
    fi
}

# Function to install Python
install_python() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y python3.12 python3.12-tk
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Assumes Homebrew is installed
        brew install python@3.12
        brew install tcl-tk
    elif [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "Please install Python 3.12 with Tkinter manually from https://www.python.org/downloads/"
        exit 1
    else
        echo "Unsupported OS for automatic Python installation."
        exit 1
    fi
}

# Function to check if Poetry is installed
check_poetry() {
    if ! command -v poetry &> /dev/null; then
        echo "Poetry is not installed. Installing Poetry..."
        install_poetry
    else
        echo "Poetry is already installed at $(which poetry)."
    fi
}

# Function to install Poetry
install_poetry() {
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> $HOME/.bashrc
}

# Function to install project dependencies
install_dependencies() {
    echo "Installing dependencies..."
    poetry install
    check_virtual_environment
}

# Function to check and manage the virtual environment
check_virtual_environment() {
    local VENV_PATH=$(poetry env info -p 2>/dev/null)
    local VENV_PYTHON_VERSION=$("$VENV_PATH/bin/python" --version 2>&1)
    local SYSTEM_PYTHON_VERSION=$(python3 --version 2>&1)

    if [[ -d "$VENV_PATH" ]]; then
        if [[ "$VENV_PYTHON_VERSION" == "$SYSTEM_PYTHON_VERSION" ]]; then
            echo "Virtual environment already exists and is using the correct Python version."
        else
            echo "Virtual environment exists but uses a different Python version. Recreating..."
            poetry env remove python
            poetry env use python3
            poetry install
        fi
    else
        echo "Creating virtual environment with the current Python version..."
        poetry env use python3
        poetry install
    fi
}

# Function to run the main application
run_application() {
    read -p "Do you want to run the application now? (y/n): " decision
    if [[ "$decision" == "y" || "$decision" == "Y" ]]; then
        poetry run python main.py
    else
        echo "Setup complete. Run the application manually with: poetry run python main.py"
    fi
}

# Main installation function
main() {
    check_python
    check_poetry
    export PATH="$HOME/.local/bin:$PATH"
    install_dependencies
    run_application
}

main
