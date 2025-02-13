import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import glob
import tqdm
from tqdm.auto import tqdm
import time
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import platform
import argparse
from UNet_model import UNet, dice_loss  # Add this import
from dataset import OxfordPetDataset
from visualization import visualize_predictions, compare_all_configs

# Enable ANSI colors on Windows
if platform.system() == 'Windows':
    os.system('color')

# Alternative approach using colorama (more reliable)
# from colorama import init, Fore
# init()  # Initialize colorama
# COLORS = {
#     1: Fore.BLUE,
#     2: Fore.GREEN,
#     3: Fore.YELLOW,
#     4: Fore.RED,
#     'ENDC': Fore.RESET
# }

# Define all the constants
NUM_DASHES = 100

# ANSI color codes
COLORS = {
    1: '\033[94m',  # Blue
    2: '\033[92m',  # Green
    3: '\033[93m',  # Orange/Yellow
    4: '\033[91m',  # Red
    'ENDC': '\033[0m'  # End color
}

# ========================================================================================================
# Add these helper functions at the top with other functions
def calculate_accuracy(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    correct = (pred == target).float().sum()
    total = target.numel()
    return correct / total

# ========================================================================================================
def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Calculate validation loss
            val_loss = criterion(outputs, masks)
            total_val_loss += val_loss.item()
            
            # Calculate validation accuracy
            accuracy = calculate_accuracy(outputs, masks)
            total_val_acc += accuracy.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_val_acc / len(val_loader)
    return avg_val_loss, avg_val_acc

# ========================================================================================================
# Define the main function
if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Train UNet with optional data augmentation')
    parser.add_argument('--augment', action='store_true', default=False,
                      help='Enable data augmentation (default: False)')
    args = parser.parse_args()
    
    print("=" * NUM_DASHES)
    print(f"Starting training process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data augmentation: {'enabled' if args.augment else 'disabled'}")
    print("=" * NUM_DASHES)
    
    # Use the flag in the code
    use_augmentation = args.augment
    
    # Device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * NUM_DASHES)
    
    # Dataset loading
    print("\nPreparing dataset...")
    
    # Base transforms that are always applied
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Augmentation transforms
    if use_augmentation:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        print("Data augmentation enabled")
    else:
        transform = base_transform
        print("Data augmentation disabled")

    # Validation always uses base transform without augmentation
    val_transform = base_transform
    
    print(f"Current working directory: {os.getcwd()}")
    dataset_path = os.path.normpath(os.path.join(".", "data", "oxford_pet"))
    print(f"Looking for dataset in: {os.path.abspath(dataset_path)}")
    
    try:
        # Create dataset once and then split it
        dataset = OxfordPetDataset(dataset_path, transform, use_augmentation=args.augment)
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create DataLoaders
        batch_size = 8
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print("=" * NUM_DASHES)
        
        configs = [
            (True, True, nn.BCEWithLogitsLoss(), "MaxPool + TransConv + BCE"),
            (True, True, dice_loss, "MaxPool + TransConv + Dice"),
            (False, True, nn.BCEWithLogitsLoss(), "StrideConv + TransConv + BCE"),
            (False, False, dice_loss, "StrideConv + Upsample + Dice")
        ]
        
        print("\nStarting training now for all configurations...")
        print("-" * NUM_DASHES)

        # Training loop for each configuration
        for config_idx, (use_maxpool, use_transpose, criterion, config_name) in enumerate(configs, 1):
            color = COLORS[config_idx]
            endc = COLORS['ENDC']
            print(f"\nTraining {color}Configuration {config_idx}/4: {config_name}{endc}")
            print("-" * NUM_DASHES)
            
            # Initialize model
            model = UNet(3, 1, use_maxpool, use_transpose).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training information
            num_epochs = 20
            num_batches = len(train_loader)
            total_steps = num_epochs * num_batches
            
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Training steps per epoch: {num_batches}")
            print(f"Total training steps: {total_steps:,}")
            
            # Create epoch progress bar
            epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
            
            # Training metrics
            best_loss = float('inf')
            start_time = time.time()
            
            for epoch in epoch_pbar:
                model.train()
                total_loss = 0
                total_acc = 0
                
                # Training loop
                batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                                leave=False, position=1)
                
                for batch_idx, (images, masks) in enumerate(batch_pbar):
                    images, masks = images.to(device), masks.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    # Calculate training accuracy
                    accuracy = calculate_accuracy(outputs, masks)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_acc += accuracy.item()
                    current_loss = total_loss / (batch_idx + 1)
                    current_acc = total_acc / (batch_idx + 1)
                    
                    # Update batch progress bar
                    batch_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{accuracy.item():.4f}',
                        'avg_loss': f'{current_loss:.4f}',
                        'avg_acc': f'{current_acc:.4f}'
                    })
                
                # Calculate training metrics
                train_loss = total_loss / len(train_loader)
                train_acc = total_acc / len(train_loader)
                
                # Calculate validation metrics
                val_loss, val_acc = validate_model(model, val_loader, criterion, device)
                
                # Update best metrics
                best_loss = min(best_loss, val_loss)
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'train_acc': f'{train_acc:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'best_loss': f'{best_loss:.4f}'
                })
                
                # Optional: Print detailed metrics after each epoch
                # print(f"\nEpoch {epoch+1}/{num_epochs}:")
                # print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                # print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # After training, visualize predictions with metrics
            print("\nGenerating prediction visualizations...")
            metrics = {
                'loss': train_loss,
                'acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            fig = visualize_predictions(model, val_loader, device, metrics=metrics)
            
            # Save the figure
            os.makedirs('results', exist_ok=True)
            fig_path = f'results/predictions_config_{config_idx}.png'
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"Saved predictions visualization to {fig_path}")
            
            # Training summary
            training_time = time.time() - start_time
            print(f"\nTraining completed for {color}{config_name}{endc}:: Best loss: {color}{best_loss:.4f}{endc}")
            print(f"Training time: {training_time:.2f} seconds")
            print("=" * NUM_DASHES)
        
        print(f"\nAll configurations completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Prediction visualizations saved in the 'results' directory")
        
        # Add comparison of all configurations
        print("\nGenerating comparison of all configurations...")
        compare_all_configs(configs)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nExpected dataset structure:")
        print("""
data/
  oxford_pet/
    images/
      image1.jpg
      image2.jpg
      ...
    annotations/
      trimaps/
        image1.png
        image2.png
        ...
        """)
        raise
