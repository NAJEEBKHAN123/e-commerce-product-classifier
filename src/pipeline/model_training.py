"""
E-commerce Product Classifier Training Script
Optimized for Colab GPU training with proper hyperparameters
"""

import multiprocessing
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

print(f"📁 Project root: {project_root}")

# Set matplotlib backend to avoid GUI issues
try:
    import matplotlib
    matplotlib.use('Agg')
    print("✅ Set matplotlib to Agg backend")
except ImportError:
    print("⚠️  matplotlib not available")

import torch
from src.data.loader import create_dataloaders
from src.model.cnn import ProductCNN
from src.model.train import Trainer
from src.model.evaluation import Evaluator
import wandb
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main training function."""
    # ========== TRAINING CONFIGURATION ==========
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    PATIENCE = 5
    
    # Data path - uses Google Drive path for Colab
    DATA_DIR = os.getenv("DATASET_PATH", "/content/drive/MyDrive/dataset")
    
    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========== PRINT CONFIGURATION ==========
    print("=" * 60)
    print("🛒 E-COMMERCE PRODUCT CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"📱 Device: {DEVICE}")
    print(f"📊 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🎯 Learning rate: {LEARNING_RATE}")
    print(f"📁 Data directory: {DATA_DIR}")
    print("=" * 60)
    
    # ========== DATASET VALIDATION ==========
    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset not found: {DATA_DIR}")
        return
    
    print("✅ Dataset found")
    
    # ========== WANDB INITIALIZATION ==========
    wandb_api_key = os.getenv("WANDB_API_KEY")
    use_wandb = False
    
    if wandb_api_key:
        try:
            wandb.init(
                project="ecommerce-product-classifier",
                config={
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "device": DEVICE,
                    "model": "ProductCNN",
                    "optimizer": "Adam",
                    "loss": "CrossEntropyLoss"
                },
                name=f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            use_wandb = True
            print("✅ WandB initialized")
        except Exception as e:
            print(f"⚠️  WandB initialization failed: {e}")
            use_wandb = False
    else:
        print("ℹ️  Running without WandB")
    
    # ========== CREATE DATALOADERS ==========
    print("\n📊 Creating dataloaders...")
    try:
        if os.name == 'nt':
            num_workers = 0
            print("   Windows detected: Using 0 workers")
        else:
            num_workers = min(4, multiprocessing.cpu_count() // 2)
            print(f"   Linux/Colab: Using {num_workers} workers")
        
        train_loader, val_loader, _, class_weights, categories = create_dataloaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=num_workers
        )
        
        print(f"✅ Categories ({len(categories)}): {categories}")
        print(f"✅ Train: {len(train_loader.dataset)} images")
        print(f"✅ Validation: {len(val_loader.dataset)} images")
        
        if class_weights is not None:
            class_weights = class_weights.to(DEVICE)
            print(f"✅ Using class weights")
            
    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        return
    
    # ========== INITIALIZE MODEL ==========
    print("\n🤖 Initializing model...")
    model = ProductCNN(num_classes=len(categories)).to(DEVICE)
    print(f"✅ Model initialized on {DEVICE}")
    
    # ========== INITIALIZE TRAINER & EVALUATOR ==========
    trainer = Trainer(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        data_loader=train_loader,
        model=model,
        model_path="models/ecommerce_cnn",
        device=DEVICE,
        class_weights=class_weights
    )
    
    evaluator = Evaluator(
        batch_size=BATCH_SIZE,
        data_loader=val_loader,
        model=model,
        device=DEVICE
    )
    
    best_accuracy = 0.0
    no_improvement_count = 0
    
    # ========== TRAINING LOOP ==========
    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING - {EPOCHS} EPOCHS")
    print(f"{'='*60}")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*40}")
        print(f"📅 Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*40}")
        
        # Training phase
        print("\n🎯 Training...")
        train_loss, train_acc = trainer.start_training_loop(epoch + 1)
        
        if train_loss is None:
            print("❌ Training failed")
            break
        
        # Validation phase
        print("\n📊 Validation...")
        val_results = evaluator.start_evaluation_loop(epoch + 1)
        
        if val_results:
            val_loss = val_results['average_loss']
            val_acc = val_results['accuracy']
            
            print(f"\n📈 Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                })
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                trainer.save_model(epoch=epoch + 1, accuracy=val_acc)
                print(f"💾 New best model: {val_acc:.2f}%")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(f"⏳ No improvement: {no_improvement_count}/{PATIENCE}")
                
                if no_improvement_count >= PATIENCE:
                    print(f"\n🛑 Early stopping")
                    break
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"🏆 Best accuracy: {best_accuracy:.2f}%")
    
    trainer.save_model(epoch=EPOCHS, accuracy=best_accuracy, final=True)
    
    if use_wandb:
        wandb.finish()
    
    return best_accuracy


if __name__ == "__main__":
    if os.name == 'nt':
        multiprocessing.freeze_support()
    
    try:
        best_acc = main()
        if best_acc is not None:
            print(f"\n✨ Training finished! Best accuracy: {best_acc:.2f}%")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()