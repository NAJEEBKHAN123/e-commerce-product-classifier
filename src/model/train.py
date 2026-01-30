# src/model/train.py - UPDATED FOR COLAB WITH SGD
import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np

class Trainer:
    """Training class for model training - Using SGD for stability"""
    
    def __init__(self, batch_size, learning_rate, data_loader, model, 
                 model_path, device, class_weights=None, patience=5):
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.model = model
        self.lr = learning_rate
        self.device = device
        self.patience = patience
        
        # Use class weights for imbalance
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            class_weights = class_weights.to(device)
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # USE SGD FOR STABILITY (not Adam)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=self.patience, 
            factor=0.5,
            verbose=False  # No verbose for Colab compatibility
        )
        
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = model_path
        
        # Track training history
        self.train_loss_history = []
        self.train_acc_history = []
        self.current_lr = self.lr
    
    def start_training_loop(self, epoch):
        """Run training for one epoch"""
        try:
            self.model.train()  # Set model to training mode
            epoch_start_time = time.time()
            
            total_loss = 0
            correct = 0
            total = 0
            
            # Store batch losses for learning rate scheduler
            batch_losses = []
            
            for batch_idx, batch_data in enumerate(self.data_loader):
                batch_start_time = time.time()
                
                # Handle different data loader formats
                if len(batch_data) == 3:
                    images, labels, _ = batch_data
                elif len(batch_data) == 2:
                    images, labels = batch_data
                else:
                    print(f"❌ Unexpected batch format: {len(batch_data)} elements")
                    continue
                
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # CRITICAL: Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                # Calculate metrics
                total_loss += loss.item()
                batch_losses.append(loss.item())
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    batch_time = time.time() - batch_start_time
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.data_loader)}: "
                          f"Loss = {loss.item():.4f}, LR = {current_lr:.6f}, Time = {batch_time:.2f}s")
            
            # Calculate epoch metrics
            avg_loss = total_loss / len(self.data_loader)
            accuracy = 100.0 * correct / total
            epoch_time = time.time() - epoch_start_time
            
            # Update learning rate scheduler based on validation loss
            self.scheduler.step(avg_loss)
            self.current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.train_loss_history.append(avg_loss)
            self.train_acc_history.append(accuracy)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Training Loss: {avg_loss:.4f}")
            print(f"  Training Accuracy: {accuracy:.2f}%")
            print(f"  Learning Rate: {self.current_lr:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f} seconds")
            print(f"  Correct: {correct}/{total}")
            
            return avg_loss, accuracy
            
        except Exception as e:
            print(f"Error in training loop at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_model(self, epoch=None, accuracy=None, final=False):
        """Save model checkpoint"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            if final:
                filename = f"{self.model_path}_final.pth"
            elif epoch is not None and accuracy is not None:
                filename = f"{self.model_path}_epoch{epoch}_acc{accuracy:.2f}.pth"
            else:
                filename = f"{self.model_path}.pth"
            
            final_path = os.path.join(self.model_dir, filename)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.train_loss_history[-1] if self.train_loss_history else 0,
                'accuracy': self.train_acc_history[-1] if self.train_acc_history else 0,
                'train_loss_history': self.train_loss_history,
                'train_acc_history': self.train_acc_history,
                'learning_rate': self.current_lr,
                'config': {
                    'batch_size': self.batch_size,
                    'initial_lr': self.lr,
                    'device': str(self.device),
                    'optimizer': 'SGD',
                    'momentum': 0.9,
                    'weight_decay': 1e-4
                }
            }
            
            torch.save(checkpoint, final_path)
            
            print(f"Model saved to: {final_path}")
            return final_path
            
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_model(self, model_path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'train_loss_history' in checkpoint:
                self.train_loss_history = checkpoint['train_loss_history']
            if 'train_acc_history' in checkpoint:
                self.train_acc_history = checkpoint['train_acc_history']
            
            if 'learning_rate' in checkpoint:
                self.current_lr = checkpoint['learning_rate']
            
            print(f"Model loaded from: {model_path}")
            print(f"Previous training: {len(self.train_loss_history)} epochs")
            if self.train_loss_history:
                print(f"Best loss: {min(self.train_loss_history):.4f}")
            if self.train_acc_history:
                print(f"Best accuracy: {max(self.train_acc_history):.2f}%")
            
            return checkpoint.get('epoch', 0)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def get_training_history(self):
        """Get training history for plotting"""
        return {
            'loss': self.train_loss_history,
            'accuracy': self.train_acc_history,
            'learning_rate': self.current_lr
        }
    
    def get_current_metrics(self):
        """Get current training metrics"""
        if self.train_loss_history and self.train_acc_history:
            return {
                'loss': self.train_loss_history[-1],
                'accuracy': self.train_acc_history[-1],
                'learning_rate': self.current_lr
            }
        return None
    
    def print_config(self):
        """Print training configuration"""
        print("\n📋 Training Configuration:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Initial learning rate: {self.lr}")
        print(f"  Optimizer: SGD with momentum=0.9")
        print(f"  Weight decay: 1e-4")
        print(f"  Gradient clipping: max_norm=0.5")
        print(f"  Device: {self.device}")
        print(f"  Scheduler: ReduceLROnPlateau (patience={self.patience})")