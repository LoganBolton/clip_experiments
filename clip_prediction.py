import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import os
import urllib.request
import zipfile
import sys

# Global variable to store the clip module
clip_module = None

def download_clip():
    """Download and setup CLIP if not already available."""
    if not os.path.exists("clip"):
        print("Downloading CLIP...")
        url = "https://github.com/openai/CLIP/archive/refs/heads/main.zip"
        urllib.request.urlretrieve(url, "clip.zip")
        
        with zipfile.ZipFile("clip.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.rename("CLIP-main", "clip")
        os.remove("clip.zip")
        print("CLIP downloaded successfully!")

def load_clip_model():
    """Load CLIP model and preprocessing function."""
    global clip_module
    
    # Download CLIP if needed
    download_clip()
    
    # Add CLIP to path and import
    if "clip" not in sys.path:
        sys.path.append("clip")
    
    # Import clip module
    import clip
    clip_module = clip
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def get_imagenet_classes():
    """Get ImageNet class names."""
    # You can also use a more comprehensive list of classes
    # For now, using a subset of common ImageNet classes
    classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    return classes

def predict_classes(image_path, model, preprocess, device, classes, top_k=10):
    """Predict classes for an image using CLIP."""
    global clip_module
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Prepare text inputs
    text_inputs = torch.cat([clip_module.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    # Get top predictions
    values, indices = similarity[0].topk(top_k)
    
    predictions = []
    for value, idx in zip(values, indices):
        predictions.append({
            'class': classes[idx],
            'confidence': value.item()
        })
    
    return predictions

def main():
    """Main function to run CLIP predictions."""
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, device = load_clip_model()
    
    # Get class names
    classes = get_imagenet_classes()
    print(f"Loaded {len(classes)} classes")
    
    # Image path
    image_path = Path("images/multi_object.png")
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Predict classes
    predictions = predict_classes(image_path, model, preprocess, device, classes, top_k=15)
    
    # Display results
    print("\nTop predictions:")
    print("-" * 50)
    for i, pred in enumerate(predictions, 1):
        confidence_percent = pred['confidence'] * 100
        print(f"{i:2d}. {pred['class']:<20} {confidence_percent:6.2f}%")

if __name__ == "__main__":
    main() 