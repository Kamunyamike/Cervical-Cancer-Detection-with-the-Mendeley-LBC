# inference.py
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = [
    "Negative for Intraepithelial malignancy",
    "Low squamous intra-epithelial lesion",
    "High squamous intra-epithelial lesion",
    "Squamous cell carcinoma",
]

IMG_SIZE = 128

# Match your notebook's val_transform
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def preprocess_image(pil_image):
    """Convert PIL to RGB np.array and apply albumentations normalization."""
    img = np.array(pil_image.convert("RGB"))
    augmented = val_transform(image=img)
    tensor = augmented["image"]  # shape: (3, H, W)
    return tensor

def predict(model, device, pil_image):
    """Return logits, probabilities, and predicted class name."""
    model.eval()
    with torch.no_grad():
        x = preprocess_image(pil_image).unsqueeze(0).to(device)  # (1,3,128,128)
        logits = model(x)  # (1,4)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (4,)
        pred_idx = int(np.argmax(probs))
        return logits.cpu().numpy()[0], probs, CLASS_NAMES[pred_idx]