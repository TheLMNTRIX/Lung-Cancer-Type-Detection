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
    learning_rate = 0.001
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms - keeping the same as they worked well
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

def create_model():
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:  # Keep last few layers trainable
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),  # Slightly higher dropout for ResNet
        nn.Linear(512, Config.num_classes)
    )
    
    return model

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
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ResNet50')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(save_dir, 'resnet_confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision = np.mean(precision)
    overall_recall = np.mean(recall)
    overall_f1 = np.mean(f1)
    
    results = {
        'per_class_metrics': per_class_accuracy,
        'overall_metrics': {
            'accuracy': float(overall_accuracy),
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1': float(overall_f1)
        }
    }
    
    metrics_path = os.path.join(save_dir, 'resnet_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def save_model_with_metadata(model, metrics, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_filename = f'lung_cancer_resnet_{timestamp}.pth'
    model_path = os.path.join(save_dir, model_filename)
    
    metadata = {
        'timestamp': timestamp,
        'architecture': 'resnet50',
        'num_classes': Config.num_classes,
        'metrics': metrics,
        'model_filename': model_filename
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, model_path)
    
    metadata_path = os.path.join(save_dir, f'resnet_model_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return model_path, metadata_path

def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['metadata']

def main():
    print("Loading data...")
    dataloaders, image_datasets = load_data()
    
    print("Creating ResNet50 model...")
    model = create_model()
    model = model.to(Config.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': list(model.fc.parameters()), 'lr': Config.learning_rate},
        {'params': list(model.layer4.parameters()), 'lr': Config.learning_rate * 0.1},
        {'params': list(model.layer3.parameters()), 'lr': Config.learning_rate * 0.01}
    ])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    print("Training model...")
    model = train_model(model, dataloaders, criterion, optimizer, scheduler)
    
    print("\nEvaluating on test set:")
    metrics = evaluate_model(model, dataloaders['test'])
    
    print("\nPer-Class Performance:")
    for class_name, class_metrics in metrics['per_class_metrics'].items():
        print(f"\n{class_name}:")
        print(f"Correct predictions: {class_metrics['correct']}/{class_metrics['total']}")
        print(f"Accuracy: {class_metrics['accuracy']:.4f}")
        print(f"Precision: {class_metrics['precision']:.4f}")
        print(f"Recall: {class_metrics['recall']:.4f}")
        print(f"F1-Score: {class_metrics['f1']:.4f}")
    
    print("\nOverall Metrics:")
    for metric, value in metrics['overall_metrics'].items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    model_path, metadata_path = save_model_with_metadata(model, metrics)
    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == '__main__':
    main()