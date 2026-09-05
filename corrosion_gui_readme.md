# Surface Corrosion Detection GUI

A desktop GUI application for surface defect/corrosion prediction using a trained CNN model.

## Features
- Load and predict on a single image
- Displays prediction class and confidence
- Visual feedback with class-specific colors

## Install
```bash
pip install -r corrosion_gui_requirements.txt
```

## Run
```bash
python corrosion_gui.py
```

## Usage
1. Click `Open Image` to select a surface image.
2. Click `Predict` to classify the defect type.
3. View the result in the status bar and info label.

## Model
- Uses the trained ResNet-50 model from `models/checkpoints/best_model.pth`
- Classifies into 6 surface defect types: Crazing, Inclusion, Patches, Pitted_surface, Rolled-in_scale, Scratches
- These defects include corrosion-related surface issues.

## Notes
- Ensure the model checkpoint exists at `models/checkpoints/best_model.pth`
- Images are resized to 224x224 for prediction
- Requires PyTorch and related libraries