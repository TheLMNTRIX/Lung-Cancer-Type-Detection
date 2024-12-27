import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime
from torch.nn import functional as F

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration
class Config:
    windows_input_path = r'C:\Users\rohit\OneDrive\Desktop\Data\PreProcessed2' # Update this
    data_dir = f"/mnt/{windows_input_path[0].lower()}{windows_input_path[2:].replace('\\', '/')}"  
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0005  # Reduced learning rate for SE blocks
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Squeeze and Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Modified DenseNet with SE blocks
class DenseNetSE(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseNetSE, self).__init__()
        # Load pretrained DenseNet
        densenet = models.densenet121(pretrained=True)
        
        # Get the features (all layers except the classifier)
        self.features = densenet.features
        
        # Add SE blocks after each dense block
        self.se1 = SEBlock(256)    # After first dense block
        self.se2 = SEBlock(512)    # After second dense block
        self.se3 = SEBlock(1024)   # After third dense block
        self.se4 = SEBlock(1024)   # After fourth dense block
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # First dense block
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        x = self.features.denseblock1(x)
        x = self.se1(x)
        x = self.features.transition1(x)
        
        # Second dense block
        x = self.features.denseblock2(x)
        x = self.se2(x)
        x = self.features.transition2(x)
        
        # Third dense block
        x = self.features.denseblock3(x)
        x = self.se3(x)
        x = self.features.transition3(x)
        
        # Fourth dense block
        x = self.features.denseblock4(x)
        x = self.se4(x)
        x = self.features.norm5(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# Load datasets
def load_data():
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(Config.data_dir, x), data_transforms[x])
        for x in ['train', 'valid', 'test']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=Config.batch_size, shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'valid', 'test']
    }
    
    return dataloaders, image_datasets

# Create model
def create_model():
    model = DenseNetSE(num_classes=Config.num_classes)
    return model

# Training function
def train_model(model, dataloaders, criterion, optimizer, scheduler):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(Config.num_epochs):
        print(f'Epoch {epoch+1}/{Config.num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
        print()
    
    model.load_state_dict(best_model_wts)
    return model

# Evaluation function (same as original)
def evaluate_model(model, dataloader, save_dir='results'):
    model.eval()
    all_preds = []
    all_labels = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(Config.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    class_names = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']
    per_class_accuracy = {}
    
    for i, class_name in enumerate(class_names):
        class_mask = (all_labels == i)
        class_correct = np.sum((all_preds == i) & class_mask)
        class_total = np.sum(class_mask)
        accuracy = class_correct / class_total if class_total > 0 else 0
        per_class_accuracy[class_name] = {
            'correct': int(class_correct),
            'total': int(class_total),
            'accuracy': float(accuracy),
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i])
        }
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    overall_metrics = {
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1': float(np.mean(f1)),
        'accuracy': float(accuracy_score(all_labels, all_preds))
    }
    
    results = {
        'per_class_metrics': per_class_accuracy,
        'overall_metrics': overall_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    metrics_path = os.path.join(save_dir, 'dense_net_se_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def save_model_with_metadata(model, metrics, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_filename = f'lung_cancer_densenet_se_{timestamp}.pth'
    model_path = os.path.join(save_dir, model_filename)
    
    metadata = {
        'timestamp': timestamp,
        'architecture': 'densenet121_se',
        'num_classes': Config.num_classes,
        'metrics': metrics,
        'model_filename': model_filename
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, model_path)
    
    metadata_path = os.path.join(save_dir, f'densenet_se_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return model_path, metadata_path

def load_model(model_path):
    checkpoint = torch.load(model_path)
    
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['metadata']

def main():
    # Load data
    dataloaders, image_datasets = load_data()
    
    # Create model
    model = create_model()
    model = model.to(Config.device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler)
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_metrics = evaluate_model(model, dataloaders['test'])
    
    # Print detailed results
    print("\nPer-Class Performance:")
    for class_name, class_metrics in test_metrics['per_class_metrics'].items():
        print(f"\n{class_name}:")
        print(f"Correct predictions: {class_metrics['correct']}/{class_metrics['total']}")
        print(f"Accuracy: {class_metrics['accuracy']:.4f}")
        print(f"Precision: {class_metrics['precision']:.4f}")
        print(f"Recall: {class_metrics['recall']:.4f}")
        print(f"F1-Score: {class_metrics['f1']:.4f}")
    
    # Calculate and print overall metrics
    print("\nOverall Metrics:")
    print(f"Accuracy: {test_metrics['overall_metrics']['accuracy']:.4f}")
    print(f"Precision: {test_metrics['overall_metrics']['precision']:.4f}")
    print(f"Recall: {test_metrics['overall_metrics']['recall']:.4f}")
    print(f"F1-Score: {test_metrics['overall_metrics']['f1']:.4f}")
    
    # Save model with metadata
    model_path, metadata_path = save_model_with_metadata(model, test_metrics)
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == '__main__':
    main()