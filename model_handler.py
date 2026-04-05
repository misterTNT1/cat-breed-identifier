import os

import torch
import torchvision
from PIL import Image

from model import Model


def get_class_names():
    path = "training_new/train"
    return [name for name in os.listdir(path)
           if os.path.isdir(os.path.join(path, name))]


# load the best model
def load_model(device, path="checkpoints/best_checkpoint.pth", model=Model(num_classes=len(get_class_names()))):
    state_dict = torch.load(path, weights_only=True)
    model = model.to(device)
    model.load_state_dict(state_dict)

    return model

image_size = (224, 224)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
])

def predict_cat_breed(cat_model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = cat_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class_idx = torch.argmax(outputs, dim=1).item()

    return get_class_names()[predicted_class_idx]

if __name__ == '__main__':
    image_path = "test_siamese.jpg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_model = load_model(device)
    cat_model.eval()
    breed = predict_cat_breed(cat_model, image_path, device)
    print(breed)