import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration class
class Config:
    data_dir = f"/home/rinzler/dev/Data/PreProcessed2"  
    batch_size = 32
    num_epochs = 2
    learning_rate = 0.0005
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),  # Ensure images are large enough for Inception v3
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load the pretrained Inception v3 model
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

# Modify the final fully connected layer to match the number of classes in your dataset
# Inception v3 has an auxiliary output during training, so we need to modify that as well
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, Config.num_classes)
)

# Modify the auxiliary output layer
num_aux_features = model.AuxLogits.fc.in_features
model.AuxLogits.fc = nn.Sequential(
    nn.Linear(num_aux_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, Config.num_classes)
)

# Move the model to the specified device
model = model.to(Config.device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=1e-4)

# Load datasets using the provided data directory and transformations
data_dir = Config.data_dir

train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
valid_dataset = ImageFolder(os.path.join(data_dir, 'valid'), transform=data_transforms['valid'])
test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

# Training and validation loop
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # For Inception v3, we need to handle auxiliary outputs
            if isinstance(outputs, tuple):
                outputs, aux_outputs = outputs
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train

        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}")

        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader):
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)

                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs

                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_valid += (preds == labels).sum().item()
                total_valid += labels.size(0)

        epoch_valid_loss = valid_loss / total_valid
        epoch_valid_acc = correct_valid / total_valid

        print(f"Valid Loss: {epoch_valid_loss:.4f}, Valid Accuracy: {epoch_valid_acc:.4f}")

    print("Training complete")
    
    # Save the trained model
    model_path = f"inception_v3_lung_ct.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Evaluate the model on test data
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load(f"inception_v3_lung_ct.pth", weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs, _ = outputs

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate accuracy for this batch
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

    # Compute confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

    print("Classification Report:\n", cr)

    # Calculate total accuracy
    total_accuracy = correct_preds / total_preds
    print(f"Total Test Accuracy: {total_accuracy:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(14,14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.title('Confusion Matrix')
    plt.savefig(f"confusion_matrix.png")
    plt.close()

# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer, Config.num_epochs)

# Evaluate the model
evaluate_model(model, test_loader)