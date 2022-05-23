#!/bin/bash
set -e  # Exit on first failure

curl -sSL https://install.python-poetry.org | python3 -

echo 'Append to your .zshrc / .bashrc or run: export PATH="$HOME/.poetry/bin:$PATH"'
