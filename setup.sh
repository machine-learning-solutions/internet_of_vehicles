#!/usr/bin/env bash

# Function to ensure pyenv is installed and initialized
ensure_pyenv() {
    if ! command -v pyenv &> /dev/null; then
        echo "pyenv is not installed. Installing pyenv..."
        install_pyenv
    else
        echo "pyenv is already installed."
    fi

    # Initialize pyenv
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    if command -v pyenv &> /dev/null; then
        eval "$(pyenv init -)"
    fi
}

# Function to install pyenv
install_pyenv() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Install pyenv on macOS using Homebrew
        brew update
        brew install pyenv
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Install pyenv on Linux
        curl https://pyenv.run | bash
    else
        echo "Unsupported OS for automatic pyenv installation."
        exit 1
    fi

    # Add pyenv to shell configuration
    if [[ -n "$BASH_VERSION" ]]; then
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$HOME/.bashrc"
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$HOME/.bashrc"
        echo 'eval "$(pyenv init -)"' >> "$HOME/.bashrc"
    elif [[ -n "$ZSH_VERSION" ]]; then
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$HOME/.zshrc"
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$HOME/.zshrc"
        echo 'eval "$(pyenv init -)"' >> "$HOME/.zshrc"
    fi
}

# Function to check if Python 3.11.10 is installed via pyenv
check_python() {
    ensure_pyenv

    if ! pyenv versions --bare | grep -q "^3.11.10$"; then
        echo "Python 3.11.10 is not installed via pyenv. Installing..."
        pyenv install 3.11.10
    else
        echo "Python 3.11.10 is already installed via pyenv."
    fi

    # Set Python 3.11.10 as the local version
    pyenv local 3.11.10

    # Ensure that 'python3' refers to pyenv's Python 3.11.10
    if [[ "$(python3 --version 2>&1)" != "Python 3.11.10" ]]; then
        echo "Setting pyenv's Python 3.11.10 as the default 'python3'"
        ln -sf "$(pyenv which python)" "$PYENV_ROOT/shims/python3"
    fi
}

# Function to check if Git is installed
check_git() {
    if ! command -v git &> /dev/null; then
        echo "Git is not installed. Installing Git..."
        install_git
    else
        echo "Git is already installed at $(which git)."
        echo "Git version: $(git --version)"
    fi
}

# Function to install Git
install_git() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Installing Git on Linux..."
        sudo apt-get update
        sudo apt-get install -y git
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installing Git on macOS..."
        brew install git
    else
        echo "Unsupported OS for automatic Git installation."
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
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    # Add to shell configuration
    if [[ -n "$BASH_VERSION" ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    elif [[ -n "$ZSH_VERSION" ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
    fi
}

# Function to install project dependencies
install_dependencies() {
    echo "Installing dependencies..."
    # Use the pyenv Python in the virtual environment
    poetry env use "$(pyenv which python)"
    poetry install
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
    check_git
    check_python
    check_poetry
    export PATH="$HOME/.local/bin:$PATH"
    install_dependencies
    run_application
}

main