# src/model/train.py
import torch
from torch.utils.data import DataLoader
import os
import time

class Trainer:
    """Training class for model training"""
    
    def __init__(self, batch_size, learning_rate, data_loader, model, 
                 model_path, device, class_weights=None):
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.model = model
        self.lr = learning_rate
        self.device = device
        
        # Use class weights for imbalance
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = model_path
        
        # Track training history
        self.train_loss_history = []
        self.train_acc_history = []
    
    def start_training_loop(self, epoch):
        """Run training for one epoch"""
        try:
            self.model.train()  # Set model to training mode
            epoch_start_time = time.time()
            
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels, _) in enumerate(self.data_loader):
                batch_start_time = time.time()
                
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Print progress
                if batch_idx % 10 == 0:
                    batch_time = time.time() - batch_start_time
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.data_loader)}: "
                          f"Loss = {loss.item():.4f}, Time = {batch_time:.2f}s")
            
            # Calculate epoch metrics
            avg_loss = total_loss / len(self.data_loader)
            accuracy = 100.0 * correct / total
            epoch_time = time.time() - epoch_start_time
            
            # Store history
            self.train_loss_history.append(avg_loss)
            self.train_acc_history.append(accuracy)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Training Loss: {avg_loss:.4f}")
            print(f"  Training Accuracy: {accuracy:.2f}%")
            print(f"  Epoch Time: {epoch_time:.2f} seconds")
            
            return avg_loss, accuracy
            
        except Exception as e:
            print(f"Error in training loop at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_model(self, epoch=None, accuracy=None):
        """Save model checkpoint"""
        try:
            if epoch is not None and accuracy is not None:
                filename = f"{self.model_path}_epoch{epoch}_acc{accuracy:.2f}.pth"
            else:
                filename = f"{self.model_path}.pth"
            
            final_path = os.path.join(self.model_dir, filename)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.train_loss_history[-1] if self.train_loss_history else 0,
                'accuracy': self.train_acc_history[-1] if self.train_acc_history else 0,
                'train_loss_history': self.train_loss_history,
                'train_acc_history': self.train_acc_history
            }, final_path)
            
            print(f"Model saved to: {final_path}")
            return final_path
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
    
    def load_model(self, model_path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'train_loss_history' in checkpoint:
                self.train_loss_history = checkpoint['train_loss_history']
            if 'train_acc_history' in checkpoint:
                self.train_acc_history = checkpoint['train_acc_history']
            
            print(f"Model loaded from: {model_path}")
            print(f"Previous training: {len(self.train_loss_history)} epochs")
            
            return checkpoint.get('epoch', 0)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return 0