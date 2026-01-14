# Real Estate Document Classifier
AI system to classify Vietnamese real estate documents (Land certificates, House ownership certificates, Contracts)

## Features

- **Document Classification**: Automatically classifies 3 types of documents
  - Sổ đỏ (Land use rights certificate)
  - Sổ hồng (House ownership certificate)
  - Hợp đồng (Purchase contract)
- **High Accuracy**: 80-90% accuracy with transfer learning
- **Web Demo**: Interactive Gradio interface
- **Legal Advice**: Provides relevant legal reminders for each document type

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/[your-username]/real_estate_classifier.git
cd real_estate_classifier

# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Train model
python train.py

# Test model
python test.py

# Run web demo
python app.py
```

## Results

- **Training Accuracy**: ~95%
- **Test Accuracy**: ~85%
- **Inference Time**: ~0.3s per image

## Tech Stack

- **PyTorch** - Deep learning framework
- **ResNet18** - Transfer learning from ImageNet
- **Gradio** - Web interface
- **PIL/Pillow** - Image processing

