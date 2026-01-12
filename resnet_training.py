import os

from torchvision.transforms import InterpolationMode

from resnet_model_head import ModelHead
import torch
import torchvision
from sklearn.metrics import classification_report, f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}.')

# Transformations
NEW_DIMENSION = 224

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),  # resize the input
    torchvision.transforms.RandomResizedCrop(
        NEW_DIMENSION,
        scale=(0.7, 1.0),
        ratio=(0.9, 1.1),
        interpolation=InterpolationMode.BILINEAR
    ),  # crop a random part of the image
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    torchvision.transforms.RandomRotation(10),  # randomly rotate the input
    torchvision.transforms.RandomPerspective(0.08, p=0.1),  # 10% chance to change the perspective (left/right/up/below)
    torchvision.transforms.RandomGrayscale(p=0.05),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

transform_valid = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# DataLoaders
TRAIN_DATA_DIR = 'training_new/train'
VALID_DATA_DIR = 'training_new/validation'
TEST_DATA_DIR = 'training_new/test'

BATCH_SIZE = 32

train_data = torchvision.datasets.ImageFolder(TRAIN_DATA_DIR,
                                              transform=transform_train,
                                              is_valid_file=lambda x: x.endswith('.jpg'))

valid_data = torchvision.datasets.ImageFolder(VALID_DATA_DIR,
                                              transform=transform_valid,
                                              is_valid_file=lambda x: x.endswith('.jpg'))

test_data = torchvision.datasets.ImageFolder(TEST_DATA_DIR,
                                             transform=transform_valid,
                                             is_valid_file=lambda x: x.endswith('.jpg'))

train_data_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_data_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# Model

# initialize model
model = torchvision.models.resnet50(pretrained=True).to(device)

# freeze the backbone
for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = ModelHead(2048, 1024, 5)
model.fc.to(device)

# Training
MODEL_SAVE_PATH = 'resnet_checkpoints'

# parameters
LEARNING_RATE = 1e-3
N_EPOCHS = 20

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(model, n_epochs, criterion, optimizer, train_data_loader, valid_data_loader,
          device, model_save_path, logging_interval: int = 50):
    best_valid_f1_score = 0.0
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(n_epochs):
        # training step
        model.train()

        for batch_idx, (batch_data, batch_labels) in enumerate(train_data_loader):
            inputs = batch_data.to(device)
            y_true = batch_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimizer step
            y_pred = model(inputs)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % logging_interval == 0:
                print(f'Epoch: {epoch + 1}\t| Batch: {batch_idx + 1}\t| Loss: {loss}')

        # validation step
        model.eval()
        y_true = []
        y_pred = []
        for valid_data, valid_labels in valid_data_loader:
            valid_data = valid_data.to(device)
            valid_labels = valid_labels.to(device)
            with torch.no_grad():
                valid_preds = model(valid_data)
            valid_pred_labels = torch.argmax(valid_preds, dim=1)
            y_true.extend(valid_labels.detach().cpu().numpy())
            y_pred.extend(valid_pred_labels.detach().cpu().numpy())
        valid_f1_score = f1_score(y_true, y_pred, average='macro')

        if valid_f1_score > best_valid_f1_score:
            best_valid_f1_score = valid_f1_score
            torch.save(model.state_dict(),
                       os.path.join(model_save_path, 'best_checkpoint.pth'))
        print(f'Epoch {epoch + 1} F1-score: {valid_f1_score}\t| Best F1-score: {best_valid_f1_score}')
        torch.save(model.state_dict(),
                   os.path.join(model_save_path, f'epoch_{epoch + 1}_checkpoint.pth'))


train(model, N_EPOCHS, criterion, optimizer,
      train_data_loader, valid_data_loader,
      device, MODEL_SAVE_PATH)

# Testing
model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, 'best_checkpoint.pth')))
model.eval()

y_true = []
y_pred = []
for test_data, test_labels in test_data_loader:
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    with torch.no_grad():
        test_preds = model(test_data)
    test_pred_labels = torch.argmax(test_preds, dim=1)
    y_true.extend(test_labels.detach().cpu().numpy())
    y_pred.extend(test_pred_labels.detach().cpu().numpy())

print(classification_report(y_true, y_pred))