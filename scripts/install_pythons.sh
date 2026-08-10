#!/usr/bin/env bash
export PYENV_ROOT=/home/akshay/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

echo "Installing Python 3.7.0 (required by thefuck/1)"
pyenv install -s 3.7.0

echo "Installing Python 3.8.20"
pyenv install -s 3.8.20

pyenv global 3.8.20
pyenv versions
