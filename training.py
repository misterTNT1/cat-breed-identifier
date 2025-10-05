from model import Model
import torch
import torchvision
from sklearn.metrics import classification_report, f1_score
import os

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
NEW_DIMENSION = 224
EARLY_STOP_INTERVAL = 5 # number of epochs in a row that are not improved for early stopping
MODEL_SAVE_PATH = "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

rotation_transform = torchvision.transforms.RandomApply([
    torchvision.transforms.RandomRotation(20),
], p=0.2)  # has a 20% chance to rotate the image between -20 and 20 degrees

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),  # resize the input
    torchvision.transforms.CenterCrop(NEW_DIMENSION),  # crop the input from the center
    torchvision.transforms.RandomPerspective(0.1, p=0.2),  # 20% chance to change the perspective (left/right/up/below)
    rotation_transform,  # randomly rotate the input (20% chance)
    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomGrayscale(p=0.1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

transform_valid = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(NEW_DIMENSION),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_dir = "training/train"
test_dir = "training/test"
validation_dir = "training/validation"

train_data = torchvision.datasets.ImageFolder(train_dir,
                                              transform=transform_train,
                                              is_valid_file=lambda x: x.endswith(".jpg"))

test_data = torchvision.datasets.ImageFolder(test_dir,
                                             transform=transform_valid,
                                             is_valid_file=lambda x: x.endswith(".jpg"))

validation_data = torchvision.datasets.ImageFolder(validation_dir,
                                                   transform=transform_valid,
                                                   is_valid_file=lambda x: x.endswith(".jpg"))

train_data_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=0)

test_data_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=0)

validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     num_workers=0)

cat_model = Model(hidden_dimension=512, num_classes=3).to(device)

# Load previous weights
checkpoint_path = os.path.join(MODEL_SAVE_PATH, "full_checkpoint.pth")
checkpoint = torch.load(checkpoint_path, weights_only=False)
# Load model weights
cat_model.load_state_dict(checkpoint['model_state_dict'])
# Recreate optimizer and load its state
optimizer = torch.optim.Adam(cat_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # same optimizer as before
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Load other info
best_f1_score = checkpoint['best_f1_score']
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)


criterion = torch.nn.CrossEntropyLoss()
cat_model.train()
def train(model, n_epochs, criterion, optimizer, train_data_loader, valid_data_loader,
          device, model_save_path, logging_interval: int = 50):
    best_f1_score = 0
    not_improved_epochs = 0
    os.makedirs(model_save_path, exist_ok=True)
    for epoch in range(n_epochs):
        model.train()
        for batch_index, (batch_data, batch_labels) in enumerate(train_data_loader):
            inputs = batch_data.to(device)
            y_true = batch_labels.to(device)

            optimizer.zero_grad()

            y_pred = model(inputs)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            if (batch_index + 1) % logging_interval == 0:
                print(f'Epoch {epoch + 1}\t| Batch: {batch_index + 1}\t| Loss: {loss.item()}')
        # validation
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

        scheduler.step(valid_f1_score)

        if valid_f1_score > best_f1_score:
            best_f1_score = valid_f1_score
            not_improved_epochs = 0
            torch.save(model.state_dict(),
                       os.path.join(model_save_path, "best_checkpoint.pth"))
        else:
            not_improved_epochs += 1
            if not_improved_epochs > EARLY_STOP_INTERVAL:
                print("Early stop triggered")
                break
        print(f'Epoch {epoch + 1} F1-score: {valid_f1_score}\t| Best F1-score: {best_f1_score}')

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1_score': best_f1_score,
            'epoch': epoch
        }, os.path.join(model_save_path, "full_checkpoint.pth"))

        torch.save(model.state_dict(),
                   os.path.join(model_save_path, f'epoch_{epoch + 1}_checkpoint.pth'))


train(cat_model, NUM_EPOCHS, criterion, optimizer,
      train_data_loader, validation_data_loader,
      device, MODEL_SAVE_PATH)

# Testing
cat_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, 'best_checkpoint.pth')))
cat_model.eval()


y_true = []
y_pred = []
for test_data, test_labels in test_data_loader:
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    with torch.no_grad():
        test_preds = cat_model(test_data)
    test_pred_labels = torch.argmax(test_preds, dim=1)
    y_true.extend(test_labels.detach().cpu().numpy())
    y_pred.extend(test_pred_labels.detach().cpu().numpy())


print(classification_report(y_true, y_pred))
