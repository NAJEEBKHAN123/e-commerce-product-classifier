import torch
import os
import time
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    """Training class with AMP support"""
    
    def __init__(self, batch_size, learning_rate, data_loader, model, 
                 model_path, device, class_weights=None, use_amp=False):
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Class weights for imbalance
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        
        # Optimizer with different learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': model.backbone.fc.parameters(), 'lr': learning_rate},
            {'params': model.backbone.layer4.parameters(), 'lr': learning_rate * 0.1}
        ])
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if self.use_amp else None
        
        # Model saving
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, model_path)
        
        # Tracking
        self.train_loss_history = []
        self.train_acc_history = []
        self.current_lr = learning_rate
        
        if self.use_amp:
            print("⚡ Mixed Precision (AMP) enabled")
    
    def start_training_loop(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_start = time.time()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _) in enumerate(self.data_loader):
            batch_start = time.time()
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress
            if batch_idx % 20 == 0:
                batch_time = time.time() - batch_start
                current_lr = self.optimizer.param_groups[0]['lr']
                acc = 100. * (predicted == labels).sum().item() / labels.size(0)
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.data_loader)}: "
                      f"Loss = {loss.item():.4f}, Acc = {acc:.2f}%, "
                      f"LR = {current_lr:.6f}, Time = {batch_time:.2f}s")
        
        # Epoch metrics
        avg_loss = total_loss / len(self.data_loader)
        accuracy = 100.0 * correct / total
        epoch_time = time.time() - epoch_start
        
        # Update scheduler
        self.scheduler.step()
        self.current_lr = self.optimizer.param_groups[0]['lr']
        
        # Store history
        self.train_loss_history.append(avg_loss)
        self.train_acc_history.append(accuracy)
        
        print(f"\n📅 Epoch {epoch} Summary:")
        print(f"  Training Loss: {avg_loss:.4f}")
        print(f"  Training Accuracy: {accuracy:.2f}%")
        print(f"  Learning Rate: {self.current_lr:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f} seconds")
        print(f"  Correct: {correct}/{total}")
        
        return avg_loss, accuracy
    
    def save_model(self, epoch, accuracy, is_final=False):
        """Save model checkpoint"""
        try:
            if is_final:
                filename = f"final_model_epoch_{epoch}_acc_{accuracy:.2f}.pth"
            else:
                filename = f"best_model_epoch_{epoch}_acc_{accuracy:.2f}.pth"
            
            save_path = os.path.join(self.model_dir, filename)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'accuracy': accuracy,
                'loss': self.train_loss_history[-1] if self.train_loss_history else 0,
                'train_loss_history': self.train_loss_history,
                'train_acc_history': self.train_acc_history,
                'learning_rate': self.current_lr,
            }, save_path)
            
            print(f"💾 Model saved: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
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
            
            print(f"✅ Model loaded: {model_path}")
            return checkpoint.get('epoch', 0)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return 0