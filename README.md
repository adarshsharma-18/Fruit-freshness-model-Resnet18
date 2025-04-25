# Fruit Type and Freshness Classification

## Overview
This project implements a deep learning model that can classify both the type of fruit and its freshness status from images. The model is built using PyTorch and is based on a modified ResNet18 architecture with multi-task learning capabilities.

## Features
- Identifies 9 different types of fruits/vegetables: apple, banana, beetroot, carrot, cucumber, orange, potato, tomato, and other
- Determines freshness status: Fresh or Spoiled
- Provides confidence scores for predictions
- Supports both PyTorch and ONNX model formats

## Model Architecture
The model uses a ResNet18 backbone pretrained on ImageNet with a custom multi-task learning head:
- Feature extraction using ResNet18
- Shared feature layers (512 → 256 → 128)
- Two separate classification heads:
  - Fruit type classification (128 → 128 → 9 classes)
  - Freshness classification (128 → 32 → 2 classes)

## Requirements
```
torch
torchvision
Pillow
onnxruntime (for ONNX model inference)
```

## Usage

### PyTorch Model
To use the PyTorch model for prediction:

```python
from test_fruit_model import predict_image

# Predict a single image
predict_image("sample/badapple.png", model_path="Fruits_edible.pt")
```

### ONNX Model
The project includes a script to convert the PyTorch model to ONNX format for deployment:

```python
# Run the conversion script
python check.py
```

This will generate a file named `Fruits_edible.onnx` that can be used with ONNX Runtime or other inference engines.

## Sample Images
The project includes sample images in the `sample` directory for testing:
- `applegood.png` - Fresh apple
- `badapple.png` - Spoiled apple
- `goodbanana.png` - Fresh banana
- `badbanana.png` - Spoiled banana
- `spoiled.png` - Generic spoiled fruit

## Model Training
The model was trained using a multi-label classification approach. The training notebook `multi-label-classification-name-freshness.ipynb` contains the complete training process.

## Example Output
```
Predicted Fruit: apple (95.23%)
Freshness: Spoiled (98.76%)
```

## License
This project is available for educational and research purposes.

## Acknowledgements
- PyTorch and torchvision for the deep learning framework and pretrained models
- ResNet architecture for the backbone feature extractor