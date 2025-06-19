# CLIP Multi-Object Image Classification

This project uses OpenAI's CLIP model to classify objects in the `multi_object.png` image.

## Setup

1. Install the required dependencies using `uv`:
```bash
uv sync
```

2. Make sure you have the `multi_object.png` image in the `images/` directory.

## Usage

Run the prediction script:
```bash
uv run python clip_prediction.py
```

The script will:
- Load the CLIP ViT-B/32 model
- Process the multi-object image
- Predict the top 15 most likely classes with confidence scores
- Display results in a formatted table

## Features

- **Automatic device detection**: Uses CUDA if available, otherwise falls back to CPU
- **Comprehensive class list**: Includes 80 common ImageNet classes
- **Confidence scoring**: Shows percentage confidence for each prediction
- **Top-k predictions**: Configurable number of top predictions to display

## Output Format

The script outputs predictions in the following format:
```
Top predictions:
--------------------------------------------------
 1. class_name               XX.XX%
 2. class_name               XX.XX%
 ...
```

## Customization

You can modify the script to:
- Use different CLIP model variants (e.g., "ViT-L/14", "RN50")
- Add custom class lists
- Change the number of top predictions
- Process multiple images

## Development

This project uses `uv` for dependency management. To add new dependencies:

```bash
uv add package-name
```

To add development dependencies:

```bash
uv add --dev package-name
``` 