#!/bin/bash

# Activate Conda env
conda activate oculr

# Then activate project-local venv (if present)
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
  echo "✅ Activated: conda 'oculr' + .venv"
else
  echo "⚠️  .venv not found — only conda env 'oculr' activated"
fi
