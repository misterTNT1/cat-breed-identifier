from datetime import datetime
import os

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import Model
from model_handler import load_model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your model
model = load_model(device)

# Load saved weights
checkpoint_path = os.path.join("checkpoints", "best_checkpoint.pth")
state_dict = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Define transforms (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5  ))
])

# Load test dataset (automatically reads from subfolders)
test_dir = "training_new/test"
test_dataset = torchvision.datasets.ImageFolder(
    test_dir,
    transform=transform,
    is_valid_file=lambda x: x.endswith(".jpg")
)

# Create a DataLoader
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# Get class names from the dataset
class_names = test_dataset.classes
print("Class names:", class_names)

# Evaluate
y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Print accuracy and detailed report
print(f"\nOverall accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%\n")
print(classification_report(y_true, y_pred, target_names=class_names))
path = "../../OneDrive\\Desktop\\confusion-matrices\\"
file_name = f"{datetime.now().strftime('%d%m_%H%M%S')}.png"
cm = confusion_matrix(y_true, y_pred)
fig = ConfusionMatrixDisplay(cm)
image = fig.plot()

image.figure_.savefig(path + file_name)
plt.show()