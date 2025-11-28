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
from .network_architectures.test_conv2d import TestConv2D
from .patch_yielder import PatchYielder, DataMode

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')


class PatchDataset(Dataset):
    """PyTorch Dataset for cached patch data."""
    
    def __init__(self, patch_yielder: PatchYielder, mode: DataMode, config: TrainingDataConfig, transform=None):
        """
        Initialize dataset.
        
        Args:
            patch_yielder: PatchYielder instance
            mode: Data mode (train/validation/test)
            config: Training data configuration
            transform: Optional data transforms
        """
        self.patch_yielder = patch_yielder
        self.mode = mode
        self.config = config
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
        
        # Apply class balancing if configured
        if self.config.equal_class_dist:
            self._balance_class_distribution()
        
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
        patch_data = self.patches[idx]
        label = self.labels[idx]
        
        # Apply zero-mean normalization if configured
        if self.config.data_with_zero_mean:
            # Apply per-channel mean subtraction for VV and VH
            normalized_data = patch_data.copy()
            for channel_idx in range(normalized_data.shape[2]):
                channel_data = normalized_data[:, :, channel_idx]
                channel_mean = np.mean(channel_data)
                normalized_data[:, :, channel_idx] = channel_data - channel_mean
            patch_data = normalized_data
        
        # Convert to tensor
        patch_tensor = torch.FloatTensor(patch_data)
        label_tensor = torch.LongTensor([label])[0]
        
        # Apply transforms if any
        if self.transform:
            patch_tensor = self.transform(patch_tensor)
        
        return patch_tensor, label_tensor
    
    def _balance_class_distribution(self):
        """Balance class distribution by subsampling to minimum class count."""
        if not self.patches or not self.labels:
            return
        
        # Count samples per class
        unique_classes, counts = np.unique(self.labels, return_counts=True)
        class_counts = dict(zip(unique_classes, counts))
        
        # Find minimum class count
        min_count = min(counts)
        logger.info(f"Balancing classes to minimum count: {min_count}")
        logger.info(f"Original class distribution: {class_counts}")
        
        # Create balanced dataset
        balanced_patches = []
        balanced_labels = []
        
        # Randomly sample from each class
        np.random.seed(42)  # For reproducibility
        
        for class_id in unique_classes:
            # Find indices for this class
            class_indices = [i for i, label in enumerate(self.labels) if label == class_id]
            
            # If we have fewer samples than min_count, we'll repeat samples
            if len(class_indices) < min_count:
                # Repeat the indices to reach min_count (with replacement)
                selected_indices = np.random.choice(class_indices, size=min_count, replace=True)
                logger.info(f"Class {class_id}: Upsampling from {len(class_indices)} to {min_count} samples")
            else:
                # Randomly select min_count samples without replacement
                selected_indices = np.random.choice(class_indices, size=min_count, replace=False)
                logger.info(f"Class {class_id}: Downsampling from {len(class_indices)} to {min_count} samples")
            
            # Add selected patches and labels
            for idx in selected_indices:
                balanced_patches.append(self.patches[idx])
                balanced_labels.append(self.labels[idx])
        
        # Replace original data with balanced data
        self.patches = balanced_patches
        self.labels = balanced_labels
        
        # Shuffle the balanced dataset
        combined = list(zip(self.patches, self.labels))
        np.random.shuffle(combined)
        self.patches, self.labels = zip(*combined)
        self.patches = list(self.patches)
        self.labels = list(self.labels)
        
        # Log new distribution
        unique_classes, counts = np.unique(self.labels, return_counts=True)
        balanced_class_counts = dict(zip(unique_classes, counts))
        logger.info(f"Balanced class distribution: {balanced_class_counts}")


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
        self.config_hash = config.get_config_hash()
        self.output_dir = project_root / "data" / "training_output" / f"run_{self.config_hash}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using config hash: {self.config_hash}")
        logger.info(f"Training output directory: {self.output_dir}")
        
        # Save complete configuration for reconstructability
        self._save_config_json(config)
        
        self.models_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.models_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize model based on architecture
        self.model = self._create_model(config).to(self.device)
        
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
    
    def _create_model(self, config: TrainingDataConfig):
        """Create model based on architecture configuration."""
        architecture_id = config.neural_network.network_architecture_id.lower()
        
        if architecture_id == 'test_flat_2_layers':
            return TestFlat2Layers(config)
        elif architecture_id == 'test_conv2d':
            return TestConv2D(config)
        else:
            logger.warning(f"Unknown architecture '{architecture_id}', defaulting to test_flat_2_layers")
            return TestFlat2Layers(config)
    
    def _save_config_json(self, config: TrainingDataConfig):
        """Save complete configuration as JSON for reconstructability."""
        from dataclasses import asdict
        import json
        from datetime import datetime
        
        # Convert config to dictionary
        config_dict = asdict(config)
        
        # Add metadata for reconstructability
        config_dict['_metadata'] = {
            'config_hash': self.config_hash,
            'generated_at': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'python_env': self._get_python_env_info()
        }
        
        # Save to JSON file
        config_file = self.output_dir / f"config_{self.config_hash}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Configuration saved to: {config_file}")
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash for reproducibility."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                capture_output=True, 
                text=True, 
                cwd=project_root,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def _get_python_env_info(self) -> dict:
        """Get Python environment information for reproducibility."""
        import sys
        import platform
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None
        except ImportError:
            torch_version = "unknown"
            cuda_available = False
            cuda_version = None
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'torch_version': torch_version,
            'cuda_available': cuda_available,
            'cuda_version': cuda_version
        }
    
    def prepare_data(self):
        """Prepare data loaders."""
        logger.info("Preparing data...")
        
        # Create patch yielder
        self.patch_yielder = PatchYielder(self.config)
        
        # Create datasets
        self.train_dataset = PatchDataset(self.patch_yielder, DataMode.TRAIN, self.config)
        self.val_dataset = PatchDataset(self.patch_yielder, DataMode.VALIDATION, self.config)
        
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
        
        return best_val_acc
    
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

    def evaluate_test_set(self) -> Tuple[float, float, Dict]:
        """Evaluate model on test set."""
        logger.info("Evaluating on test set...")
        
        # Create test dataset
        test_dataset = PatchDataset(self.patch_yielder, DataMode.TEST, self.config)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.neural_network.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Loaded {len(test_dataset)} test patches")
        
        # Evaluate
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
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
                
                # Store predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions
        
        logger.info(f"Test Results: Loss={avg_loss:.6f}, Accuracy={accuracy:.4f}")
        
        # Calculate detailed metrics
        test_metrics = {
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'accuracy': accuracy,
            'loss': avg_loss
        }
        
        return avg_loss, accuracy, test_metrics

    def generate_test_report(self, test_metrics: Dict):
        """Generate detailed test set report."""
        predictions = test_metrics['predictions']
        targets = test_metrics['targets']
        probabilities = np.array(test_metrics['probabilities'])
        
        # Map indices back to original class labels
        pred_labels = [self.idx_to_class[idx] for idx in predictions]
        target_labels = targets
        
        # Classification report
        class_names = [str(cls) for cls in self.config.classes]
        report = classification_report(
            target_labels, pred_labels,
            labels=self.config.classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Save detailed test report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_report_path = self.logs_dir / f"test_classification_report_{timestamp}.json"
        
        with open(test_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test classification report saved to: {test_report_path}")
        
        # Test confusion matrix
        cm = confusion_matrix(target_labels, pred_labels, labels=self.config.classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Test Set Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        test_cm_path = self.plots_dir / f"test_confusion_matrix_{timestamp}.png"
        plt.savefig(test_cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"Test confusion matrix saved to: {test_cm_path}")
        plt.close()
        
        return report

    def save_training_summary(self, best_val_acc: float, test_metrics: Dict, test_report: Dict):
        """Save comprehensive training summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get git information
        try:
            import subprocess
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project_root).decode().strip()
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=project_root).decode().strip()
            git_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=project_root).decode().strip()
            git_dirty = len(git_status) > 0
        except:
            git_commit = "unknown"
            git_branch = "unknown"
            git_dirty = True
        
        # Create comprehensive summary
        summary = {
            "metadata": {
                "timestamp": timestamp,
                "config_hash": self.config_hash,
                "git_commit": git_commit,
                "git_branch": git_branch,
                "git_dirty": git_dirty,
                "training_duration_seconds": getattr(self, 'training_duration', None)
            },
            "configuration": {
                "classes": self.config.classes,
                "patch_size": self.config.neural_network.patch_size,
                "batch_size": self.config.neural_network.batch_size,
                "architecture": self.config.neural_network.network_architecture_id,
                "optimizer": self.config.neural_network.optimizer,
                "dropout_rate": self.config.neural_network.dropout_rate,
                "activation_function": self.config.neural_network.activation_function,
                "layer_sizes": self.config.neural_network.layer_sizes,
                "n_epochs": self.config.neural_network.n_epochs,
                "early_stopping_patience": self.config.neural_network.early_stopping_patience,
                "data_with_zero_mean": self.config.data_with_zero_mean,
                "equal_class_dist": self.config.equal_class_dist,
                "orbits": self.config.orbits,
                "dates": self.config.dates
            },
            "training_results": {
                "epochs_completed": len(self.history['train_loss']),
                "best_validation_accuracy": best_val_acc,
                "final_train_loss": self.history['train_loss'][-1] if self.history['train_loss'] else None,
                "final_val_loss": self.history['val_loss'][-1] if self.history['val_loss'] else None,
                "final_train_accuracy": self.history['train_acc'][-1] if self.history['train_acc'] else None,
                "final_val_accuracy": self.history['val_acc'][-1] if self.history['val_acc'] else None
            },
            "test_results": {
                "test_accuracy": test_metrics['accuracy'],
                "test_loss": test_metrics['loss'],
                "per_class_metrics": {
                    str(cls): {
                        "precision": test_report[str(cls)]["precision"],
                        "recall": test_report[str(cls)]["recall"],
                        "f1_score": test_report[str(cls)]["f1-score"],
                        "support": test_report[str(cls)]["support"]
                    }
                    for cls in self.config.classes if str(cls) in test_report
                },
                "macro_avg": test_report.get("macro avg", {}),
                "weighted_avg": test_report.get("weighted avg", {})
            },
            "data_info": {
                "train_samples": len(self.train_loader.dataset),
                "validation_samples": len(self.val_loader.dataset),
                "test_samples": len(test_metrics['predictions']),
                "train_batches": len(self.train_loader),
                "validation_batches": len(self.val_loader)
            }
        }
        
        # Save summary
        summary_path = self.output_dir / f"training_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Also save a simple summary file with key metrics
        simple_summary_path = self.output_dir / "latest_results.txt"
        with open(simple_summary_path, 'w') as f:
            f.write(f"Training Summary - {timestamp}\n")
            f.write(f"Config Hash: {self.config_hash}\n")
            f.write(f"Architecture: {self.config.neural_network.network_architecture_id}\n")
            f.write(f"Zero Mean: {self.config.data_with_zero_mean}\n")
            f.write(f"Equal Class Distribution: {self.config.equal_class_dist}\n")
            f.write(f"Epochs: {len(self.history['train_loss'])}\n")
            f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
            f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Test Loss: {test_metrics['loss']:.6f}\n")
            f.write(f"\nPer-Class Test Results:\n")
            for cls in self.config.classes:
                if str(cls) in test_report:
                    metrics = test_report[str(cls)]
                    f.write(f"  Class {cls}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\n")
        
        logger.info(f"Simple summary saved to: {simple_summary_path}")
        
        return summary_path


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
        training_start = time.time()
        best_val_acc = trainer.train(save_best=True)
        trainer.training_duration = time.time() - training_start
        
        # Evaluate on test set
        logger.info("Starting test set evaluation...")
        test_loss, test_acc, test_metrics = trainer.evaluate_test_set()
        
        # Generate test report
        test_report = trainer.generate_test_report(test_metrics)
        
        # Save comprehensive summary
        summary_path = trainer.save_training_summary(best_val_acc, test_metrics, test_report)
        
        logger.info("✅ Training and evaluation completed successfully!")
        logger.info(f"📊 Summary saved to: {summary_path}")
        logger.info(f"🎯 Final Results: Val Acc={best_val_acc:.4f}, Test Acc={test_acc:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
