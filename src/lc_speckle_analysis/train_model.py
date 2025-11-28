#!/usr/bin/env python3
"""
Training script for TestFlat2Layers network architecture.
Trains on cached patch data with live validation monitoring.
"""

import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .data_config import TrainingDataConfig
from .network_architectures.test_flat_2_layers import TestFlat2Layers
from .patch_yielder import PatchYielder, DataMode

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')


class PatchDataset(Dataset):
    """PyTorch Dataset for cached patch data."""
    
    def __init__(self, patch_yielder: PatchYielder, mode: DataMode, transform=None):
        """
        Initialize dataset.
        
        Args:
            patch_yielder: PatchYielder instance
            mode: Data mode (train/validation/test)
            transform: Optional data transforms
        """
        self.patch_yielder = patch_yielder
        self.mode = mode
        self.transform = transform
        
        # Collect all patches at initialization
        logger.info(f"Loading {mode.value} dataset...")
        self.patches = []
        self.labels = []
        
        start_time = time.time()
        for patch_data, class_id in self.patch_yielder.yield_patch(mode):
            self.patches.append(patch_data)
            self.labels.append(class_id)
        
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(self.patches)} {mode.value} patches in {load_time:.2f}s")
        
        # Convert to numpy arrays for efficiency
        self.patches = np.array(self.patches)
        self.labels = np.array(self.labels)
        
        # Log class distribution
        unique_classes, counts = np.unique(self.labels, return_counts=True)
        class_dist = dict(zip(unique_classes, counts))
        logger.info(f"{mode.value} class distribution: {class_dist}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        patch_tensor = torch.FloatTensor(patch)
        label_tensor = torch.LongTensor([label])[0]
        
        # Apply transforms if any
        if self.transform:
            patch_tensor = self.transform(patch_tensor)
        
        return patch_tensor, label_tensor


class ModelTrainer:
    """Training manager for TestFlat2Layers model."""
    
    def __init__(self, config: TrainingDataConfig, device: str = 'auto'):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.config = config
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Create output directories with config hash to prevent conflicts
        config_hash = config.get_config_hash()
        self.output_dir = project_root / "data" / "training_output" / f"run_{config_hash}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using config hash: {config_hash}")
        logger.info(f"Training output directory: {self.output_dir}")
        
        self.models_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.models_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = TestFlat2Layers(config).to(self.device)
        
        # Initialize optimizer
        if config.neural_network.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        elif config.neural_network.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9)
        else:
            logger.warning(f"Unknown optimizer {config.neural_network.optimizer}, using Adam")
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        # Map config classes to model output indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(config.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def prepare_data(self):
        """Prepare data loaders."""
        logger.info("Preparing data...")
        
        # Create patch yielder
        self.patch_yielder = PatchYielder(self.config)
        
        # Create datasets
        self.train_dataset = PatchDataset(self.patch_yielder, DataMode.TRAIN)
        self.val_dataset = PatchDataset(self.patch_yielder, DataMode.VALIDATION)
        
        # Create data loaders
        batch_size = self.config.neural_network.batch_size
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Map target classes to indices
            target_mapped = torch.zeros_like(target)
            for i, cls in enumerate(target.cpu().numpy()):
                target_mapped[i] = self.class_to_idx[cls]
            target_mapped = target_mapped.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target_mapped)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct_predictions += pred.eq(target_mapped).sum().item()
            total_predictions += target.size(0)
            
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{len(self.train_loader)}: Loss {loss.item():.6f}")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Map target classes to indices
                target_mapped = torch.zeros_like(target)
                for i, cls in enumerate(target.cpu().numpy()):
                    target_mapped[i] = self.class_to_idx[cls]
                target_mapped = target_mapped.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target_mapped)
                
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct_predictions += pred.eq(target_mapped).sum().item()
                total_predictions += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        # Calculate per-class metrics
        metrics = {
            'predictions': all_predictions,
            'targets': all_targets,
            'accuracy': accuracy,
            'loss': avg_loss
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, save_best: bool = True):
        """
        Main training loop.
        
        Args:
            save_best: Whether to save the best model
        """
        epochs = self.config.neural_network.n_epochs
        patience = self.config.neural_network.early_stopping_patience
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Early stopping patience: {patience} epochs")
        
        best_val_acc = 0.0
        no_improve_epochs = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s): "
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
                logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            # Early stopping
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pth")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Generate final plots and reports
        self.plot_training_history()
        self.generate_classification_report()
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'class_mapping': {
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class
            }
        }
        
        save_path = self.models_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to: {save_path}")
    
    def plot_training_history(self):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.history['lr'], 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Final validation accuracy
        axes[1, 1].text(0.5, 0.5, f"Final Validation Accuracy:\n{self.history['val_acc'][-1]:.4f}", 
                        transform=axes[1, 1].transAxes, fontsize=16, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1, 1].set_title('Final Results')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.plots_dir / f"training_history_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to: {plot_path}")
        
        plt.close()
    
    def generate_classification_report(self):
        """Generate detailed classification report."""
        # Get final validation predictions
        val_loss, val_acc, val_metrics = self.validate()
        
        predictions = val_metrics['predictions']
        targets = val_metrics['targets']
        
        # Map indices back to original class labels
        pred_labels = [self.idx_to_class[idx] for idx in predictions]
        target_labels = targets  # These are already in original class format
        
        # Classification report
        class_names = [str(cls) for cls in self.config.classes]
        report = classification_report(
            target_labels, pred_labels, 
            labels=self.config.classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Save report as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.logs_dir / f"classification_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Classification report saved to: {report_path}")
        
        # Print summary
        logger.info("Classification Results:")
        for class_name in class_names:
            metrics = report[class_name]
            logger.info(f"  Class {class_name}: Precision={metrics['precision']:.3f}, "
                       f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(target_labels, pred_labels, labels=self.config.classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        cm_path = self.plots_dir / f"confusion_matrix_{timestamp}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {cm_path}")
        
        plt.close()


def main():
    """Main training function."""
    try:
        logger.info("Starting TestFlat2Layers training script...")
        
        # Load configuration
        config_path = project_root / "data" / "config.conf"
        config = TrainingDataConfig.from_file(config_path)
        
        # Log configuration
        logger.info("Training Configuration:")
        logger.info(f"  Classes: {config.classes}")
        logger.info(f"  Patch size: {config.neural_network.patch_size}")
        logger.info(f"  Batch size: {config.neural_network.batch_size}")
        logger.info(f"  Architecture: {config.neural_network.network_architecture_id}")
        logger.info(f"  Optimizer: {config.neural_network.optimizer}")
        
        # Create trainer
        trainer = ModelTrainer(config)
        
        # Prepare data
        trainer.prepare_data()
        
        # Train model
        trainer.train(save_best=True)
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
