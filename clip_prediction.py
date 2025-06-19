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

def get_object_classes():
    """Get a comprehensive list of object classes."""
    classes = [
        # Fruits
        "apple", "orange", "banana", "lemon", "pear", "lime", "grape",
        # Containers
        "plate", "bowl", "jar", "bottle", "cup", "mug", "glass",
        # Beverages
        "soda can", "coca cola", "pepsi", "sprite", "beer can", "energy drink",
        # Utensils and tools
        "knife", "fork", "spoon", "whisk", "scissors", "spatula", "tongs",
        # Packaged items
        "packaged snack", "oreo cookies", "chips", "candy", "chocolate",
        # Kitchen items
        "cutting board", "napkin", "paper towel", "aluminum foil",
        # General categories that might help
        "food", "drink", "tool", "container", "package"
    ]
    return classes

def method1_softmax_ranking(image_path, model, preprocess, device, classes, top_k=15):
    """
    Method 1: Traditional softmax ranking (your current approach, but improved)
    Good for: Getting the most likely classes overall
    """
    global clip_module
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Prepare text inputs with better prompts
    text_inputs = torch.cat([clip_module.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity and apply softmax
        logits = 100.0 * image_features @ text_features.T
        probabilities = logits.softmax(dim=-1)
        
    # Get top predictions
    values, indices = probabilities[0].topk(top_k)
    
    predictions = []
    for value, idx in zip(values, indices):
        predictions.append({
            'class': classes[idx],
            'probability': value.item(),
            'raw_score': logits[0][idx].item()
        })
    
    return predictions

def method2_binary_classification(image_path, model, preprocess, device, classes, threshold=0.5):
    """
    Method 2: Binary classification for each object
    Good for: Determining which objects are present/absent
    """
    global clip_module
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Get image features once
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    results = []
    
    for cls in classes:
        # Create positive and negative prompts
        positive_prompts = [
            f"a photo containing a {cls}",
            f"a picture with a {cls}",
            f"an image that has a {cls}"
        ]
        negative_prompts = [
            f"a photo without a {cls}",
            f"a picture with no {cls}",
            f"an image that has no {cls}"
        ]
        
        # Tokenize prompts
        pos_inputs = torch.cat([clip_module.tokenize(prompt) for prompt in positive_prompts]).to(device)
        neg_inputs = torch.cat([clip_module.tokenize(prompt) for prompt in negative_prompts]).to(device)
        
        with torch.no_grad():
            # Get text features
            pos_features = model.encode_text(pos_inputs)
            neg_features = model.encode_text(neg_inputs)
            
            # Normalize
            pos_features /= pos_features.norm(dim=-1, keepdim=True)
            neg_features /= neg_features.norm(dim=-1, keepdim=True)
            
            # Average the features for positive and negative
            pos_avg = pos_features.mean(dim=0, keepdim=True)
            neg_avg = neg_features.mean(dim=0, keepdim=True)
            
            # Calculate similarities
            pos_sim = (image_features @ pos_avg.T).item()
            neg_sim = (image_features @ neg_avg.T).item()
            
            # Convert to probability (positive vs negative)
            logits = torch.tensor([neg_sim, pos_sim])
            prob_present = F.softmax(logits, dim=0)[1].item()
            
        results.append({
            'class': cls,
            'probability_present': prob_present,
            'is_present': prob_present > threshold,
            'confidence': abs(prob_present - 0.5) * 2  # How confident are we?
        })
    
    # Sort by probability of being present
    results.sort(key=lambda x: x['probability_present'], reverse=True)
    
    return results

def method3_multi_label_sigmoid(image_path, model, preprocess, device, classes, threshold=0.3):
    """
    Method 3: Multi-label classification using sigmoid
    Good for: Independent probability for each class
    """
    global clip_module
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Prepare text inputs
    text_inputs = torch.cat([clip_module.tokenize(f"a photo containing a {c}") for c in classes]).to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate raw similarity scores
        similarities = (image_features @ text_features.T)[0]
        
        # Apply sigmoid to get independent probabilities
        probabilities = torch.sigmoid(similarities * 2.5)  # Scale factor for better separation
        
    results = []
    for i, cls in enumerate(classes):
        prob = probabilities[i].item()
        results.append({
            'class': cls,
            'probability': prob,
            'predicted': prob > threshold,
            'raw_similarity': similarities[i].item()
        })
    
    # Sort by probability
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return results

def compare_methods(image_path, model, preprocess, device, classes):
    """Compare all three methods and display results."""
    
    print("=" * 80)
    print("METHOD 1: SOFTMAX RANKING (Traditional Classification)")
    print("=" * 80)
    method1_results = method1_softmax_ranking(image_path, model, preprocess, device, classes, top_k=20)
    
    for i, result in enumerate(method1_results, 1):
        print(f"{i:2d}. {result['class']:<20} {result['probability']*100:6.2f}% (score: {result['raw_score']:6.2f})")
    
    print("\n" + "=" * 80)
    print("METHOD 2: BINARY CLASSIFICATION (Present/Absent)")
    print("=" * 80)
    method2_results = method2_binary_classification(image_path, model, preprocess, device, classes)
    
    # Show top candidates and definite predictions
    print("Most likely objects:")
    for result in method2_results[:20]:
        status = "✓ PRESENT" if result['is_present'] else "✗ absent"
        print(f"{result['class']:<20} {result['probability_present']*100:6.2f}% confidence {status}")
    
    # Show definite detections
    detected = [r for r in method2_results if r['is_present']]
    print(f"\nDetected objects ({len(detected)}): {', '.join([r['class'] for r in detected])}")
    
    print("\n" + "=" * 80)
    print("METHOD 3: MULTI-LABEL SIGMOID (Independent Probabilities)")
    print("=" * 80)
    method3_results = method3_multi_label_sigmoid(image_path, model, preprocess, device, classes)
    
    for i, result in enumerate(method3_results[:20], 1):
        status = "✓" if result['predicted'] else " "
        print(f"{status} {i:2d}. {result['class']:<20} {result['probability']*100:6.2f}% (sim: {result['raw_similarity']:6.2f})")
    
    # Show predictions above threshold
    predicted = [r for r in method3_results if r['predicted']]
    print(f"\nPredicted objects: {', '.join([r['class'] for r in predicted])}")

def main():
    """Main function demonstrating different multi-class approaches."""
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, device = load_clip_model()
    
    # Get class names
    classes = get_object_classes()
    print(f"Loaded {len(classes)} classes")
    
    # Image path
    image_path = Path("images/multi_object.png")
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Available classes: {', '.join(classes)}")
    
    # Compare all methods
    compare_methods(image_path, model, preprocess, device, classes)

if __name__ == "__main__":
    main()