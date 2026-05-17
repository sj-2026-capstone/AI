import os
import torch
import timm
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_PATH = os.path.join(os.path.dirname(__file__), "../models/swin_dataset_best.pth")

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=2
)

checkpoint = torch.load(BEST_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    print(f"[DEBUG] raw output: {output[0].tolist()}")
    print(f"[DEBUG] probs[0]={probs[0].item():.4f}, probs[1]={probs[1].item():.4f}")

    normal_prob = probs[0].item()
    defect_prob = probs[1].item()

    if normal_prob >= defect_prob:
        prediction = "NORMAL"
        confidence = normal_prob
    else:
        prediction = "DEFECT"
        confidence = defect_prob

    return {
        "prediction": prediction,
        "confidence": confidence,
        "normal_probability": normal_prob,
        "defect_probability": defect_prob
    }
