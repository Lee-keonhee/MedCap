#!/bin/bash
# Project 1: Medical Multimodal - Directory Setup Script

echo "Creating Project 1: Medical Multimodal Report Generation"
echo "=========================================================="

# Create directory structure
mkdir -p data/{raw,processed,external}
mkdir -p models/{checkpoints,final}
mkdir -p notebooks/{exploration,experiments}
mkdir -p src/{data,models,training,inference,utils}
mkdir -p tests
mkdir -p outputs/{reports,visualizations}
mkdir -p configs
mkdir -p docs

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# Create placeholder files
touch src/data/dataset.py
touch src/data/preprocessing.py
touch src/models/vision_encoder.py
touch src/models/language_decoder.py
touch src/models/multimodal_model.py
touch src/training/train.py
touch src/training/evaluate.py
touch src/inference/predict.py
touch src/utils/metrics.py
touch src/utils/visualization.py
touch tests/test_dataset.py
touch tests/test_model.py

echo "✅ Directory structure created successfully!"
echo ""
echo "Project Structure:"
tree -L 2 -I '__pycache__|*.pyc'
