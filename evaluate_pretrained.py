import torch
import os
import json
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from main import DenseNetSE, Config, evaluate_model, load_model  # Import necessary components from the original file

class Config:
    windows_input_path = r'C:\Users\rohit\OneDrive\Desktop\Data\PreProcessed2' # Update this to the folder with image data
    data_dir = f"/mnt/{windows_input_path[0].lower()}{windows_input_path[2:].replace('\\', '/')}"  
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0005  # Reduced learning rate for SE blocks
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data transforms for the test set
data_transforms = {
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load test data
def load_test_data():
    data_dir = Config.data_dir  # Ensure data_dir is correctly set in Config
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    return test_loader

def main():
    # Path to the pretrained model
    pretrained_model_path = "./models/lung_cancer_densenet_se_20241113_063754.pth"  # Update this with your model's path

    # Load the pretrained model
    print("Loading pretrained model...")
    model, metadata = load_model(pretrained_model_path)
    model = model.to(Config.device)
    
    print(f"Loaded model metadata: {metadata}")
    
    # Load test data
    test_loader = load_test_data()
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_metrics = evaluate_model(model, test_loader)
    
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

if __name__ == '__main__':
    main()
