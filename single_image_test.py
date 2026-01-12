import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import Model
from model_handler import get_class_names


def predict_single_image(model, image_path, class_names, device, image_size=224):
    """
    Predict the class of a single image using a trained model.

    Args:
        model (torch.nn.Module): Your trained PyTorch model.
        image_path (str): Path to the image file.
        class_names (list): List of class names in the same order as during training.
        device (torch.device): 'cuda' or 'cpu'.
        image_size (int): Resize dimension (default 224).

    Returns:
        str: Predicted class name.
    """

    # Validation-like transforms (no random augmentations)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Open the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
        predicted_class_idx = torch.argmax(outputs, dim=1).item()

    predicted_class = class_names[predicted_class_idx]
    print(f"Predicted class: {predicted_class}")
    print(f"Model confidence: {confidence * 100:.2f}%")

    print("\nClass probabilities:")
    for name, p in zip(class_names, probabilities[0]):
        print(f"  {name:<15}: {p.item() * 100:.2f}%")

    return predicted_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)
image_path = "test_siamese.jpg"
model_path = "checkpoints/best_checkpoint.pth"
cat_model = Model(512, 5).to(device)

state_dict = torch.load(model_path, weights_only=True)
cat_model.load_state_dict(state_dict)
cat_model.eval()
class_names = get_class_names()

predict_single_image(cat_model, image_path, class_names, device)
