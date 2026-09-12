"""
Smoke test: verify CLIP loads and runs on this machine (Apple M3 / MPS).

Encodes 4 candidate captions against a synthetic test image and prints
similarity scores. If this runs, the retrieval backbone is working.
"""
import torch
import clip
from PIL import Image, ImageDraw

# --- 1. Pick the best available device -------------------------------------
if torch.backends.mps.is_available():
    device = "mps"        # Apple Silicon GPU
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device: {device}")

# --- 2. Load frozen CLIP (ViT-B/32, as in the proposal) ---------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("CLIP ViT-B/32 loaded OK")

# --- 3. Make a simple synthetic image (blue square = 'sky/water' scene) -----
img = Image.new("RGB", (224, 224), (70, 130, 180))       # steel blue
draw = ImageDraw.Draw(img)
draw.ellipse([60, 60, 160, 160], fill=(255, 215, 0))     # yellow circle (sun)
image_input = preprocess(img).unsqueeze(0).to(device)

# --- 4. Candidate captions ---------------------------------------------------
captions = [
    "the sun in a blue sky",
    "a dog playing in the park",
    "a red sports car",
    "a plate of food",
]
text_input = clip.tokenize(captions).to(device)

# --- 5. Encode and rank -------------------------------------------------------
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("\nSimilarity ranking (should rank 'sun in a blue sky' first):")
for caption, score in sorted(zip(captions, similarity[0].tolist()),
                             key=lambda x: -x[1]):
    print(f"  {score:6.2%}  {caption}")

print("\nSmoke test PASSED" if similarity[0].argmax() == 0
      else "\nWARNING: unexpected top match")
