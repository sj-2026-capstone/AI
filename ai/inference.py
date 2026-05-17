import os
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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


def _swin_reshape_transform(tensor, height=7, width=7):
    if tensor.ndim == 4:
        # timm 최신 버전: (B, H, W, C) → (B, C, H, W)
        return tensor.permute(0, 3, 1, 2)
    # 구버전: (B, N, C) → (B, C, H, W)
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
    return result.permute(0, 3, 1, 2)


def generate_grad_cam(image_path: str, output_path: str, target_class_idx: int = 1):
    image = Image.open(image_path).convert("RGB")
    img_resized = image.resize((224, 224))
    rgb_img = np.array(img_resized, dtype=np.float32) / 255.0

    input_tensor = transform(image).unsqueeze(0).to(device)

    target_layers = [model.layers[-1].blocks[-1].norm1]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=_swin_reshape_transform)

    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization).save(output_path)
