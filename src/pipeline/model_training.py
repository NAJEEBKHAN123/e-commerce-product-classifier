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
    matplotlib.use('Agg')  # Use non-interactive backend
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
    try:
        # ========== TRAINING CONFIGURATION ==========
        EPOCHS = 10                    # Train for 10 epochs
        BATCH_SIZE = 64                # Increased for GPU efficiency
        LEARNING_RATE = 0.005
        PATIENCE = 5                   # Early stopping patience
        
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
            
            # Try alternative paths
            alternative_paths = [
                "d:\\ecommerce-product-classifier\\dataset",  # Windows
                "/content/dataset",                          # Colab
                "dataset",                                   # Local
                os.path.join(project_root, "dataset"),       # Project root
            ]
            
            for path in alternative_paths:
                if os.path.exists(path):
                    DATA_DIR = path
                    print(f"✅ Found dataset at: {DATA_DIR}")
                    break
            else:
                print("❌ Dataset not found in any location")
                print("\n💡 Please set DATASET_PATH environment variable or")
                print("   place dataset in one of the following locations:")
                for path in alternative_paths:
                    print(f"   - {path}")
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
                    name=f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    notes="Training on 9 product categories"
                )
                use_wandb = True
                print("✅ WandB initialized")
                print(f"🔗 View at: {wandb.run.url}")
            except Exception as e:
                print(f"⚠️  WandB initialization failed: {e}")
                use_wandb = False
        else:
            print("ℹ️  Running without WandB (set WANDB_API_KEY for logging)")
        
        # ========== CREATE DATALOADERS ==========
        print("\n📊 Creating dataloaders...")
        try:
            # Configure workers based on OS
            if os.name == 'nt':  # Windows
                num_workers = 0
                print("   Windows detected: Using 0 workers")
            else:  # Linux/Colab
                num_workers = min(4, multiprocessing.cpu_count() // 2)
                print(f"   Linux/Colab: Using {num_workers} workers")
            
            train_loader, val_loader, _, class_weights, categories = create_dataloaders(
                data_dir=DATA_DIR,
                batch_size=BATCH_SIZE,
                num_workers=num_workers
            )
            
            print(f"✅ Categories ({len(categories)}): {categories}")
            print(f"✅ Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
            print(f"✅ Validation: {len(val_loader.dataset)} images, {len(val_loader)} batches")
            
            if class_weights is not None:
                class_weights = class_weights.to(DEVICE)
                print(f"✅ Using class weights for imbalanced data")
                
        except Exception as e:
            print(f"❌ Error creating dataloaders: {e}")
            print("\n🔍 Debugging dataset structure...")
            
            # List dataset contents for debugging
            if os.path.exists(DATA_DIR):
                for split in ['train', 'val', 'check']:
                    split_path = os.path.join(DATA_DIR, split)
                    if os.path.exists(split_path):
                        print(f"\n{split.upper()} folder:")
                        categories = os.listdir(split_path)
                        print(f"  Categories: {len(categories)}")
                        for cat in categories[:5]:  # Show first 5
                            cat_path = os.path.join(split_path, cat)
                            if os.path.isdir(cat_path):
                                images = [f for f in os.listdir(cat_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                print(f"  {cat}: {len(images)} images")
                        if len(categories) > 5:
                            print(f"  ... and {len(categories) - 5} more")
            return
        
        # ========== INITIALIZE MODEL ==========
        print("\n🤖 Initializing model...")
        try:
            model = ProductCNN(num_classes=len(categories)).to(DEVICE)
            print(f"✅ Model initialized on {DEVICE}")
            print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Log model architecture to WandB
            if use_wandb:
                wandb.config.update({"total_parameters": sum(p.numel() for p in model.parameters())})
                
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            # Fallback to default 9 classes
            model = ProductCNN(num_classes=9).to(DEVICE)
            print(f"✅ Model initialized with 9 classes on {DEVICE}")
        
        # ========== INITIALIZE TRAINER & EVALUATOR ==========
        trainer = Trainer(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            data_loader=train_loader,
            model=model,
            model_path="models/ecommerce_cnn",
            device=DEVICE,
            class_weights=class_weights,
            patience=PATIENCE
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
                print("❌ Training failed, stopping...")
                break
            
            # Validation phase
            print("\n📊 Validation...")
            val_results = evaluator.start_evaluation_loop(epoch + 1)
            
            if val_results:
                val_loss = val_results['average_loss']
                val_acc = val_results['accuracy']
                
                # Print epoch summary
                print(f"\n📈 Epoch {epoch + 1} Summary:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Log to WandB
                if use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "learning_rate": trainer.optimizer.param_groups[0]['lr']
                    })
                
                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    model_path = trainer.save_model(epoch=epoch + 1, accuracy=val_acc)
                    print(f"💾 New best model saved: {val_acc:.2f}%")
                    no_improvement_count = 0
                    
                    # Log model to WandB
                    if use_wandb and model_path:
                        wandb.save(model_path)
                        wandb.run.summary["best_accuracy"] = best_accuracy
                        wandb.run.summary["best_epoch"] = epoch + 1
                else:
                    no_improvement_count += 1
                    print(f"⏳ No improvement for {no_improvement_count} epoch(s)")
                    
                    # Early stopping check
                    if no_improvement_count >= PATIENCE:
                        print(f"\n🛑 Early stopping triggered after {PATIENCE} epochs without improvement")
                        break
            else:
                print("❌ Validation failed")
        
        # ========== TRAINING COMPLETE ==========
        print(f"\n{'='*60}")
        print("🎉 TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"🏆 Best validation accuracy: {best_accuracy:.2f}%")
        
        # Save final model
        final_path = trainer.save_model(epoch=EPOCHS, accuracy=best_accuracy, final=True)
        if final_path:
            print(f"💾 Final model saved to: {final_path}")
        
        # Save training summary
        summary_path = os.path.join("models", "training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"================\n")
            f.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
            f.write(f"Total Epochs: {epoch + 1}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Learning Rate: {LEARNING_RATE}\n")
            f.write(f"Device: {DEVICE}\n")
            f.write(f"Categories: {len(categories)}\n")
            f.write(f"Categories List: {categories}\n")
        
        print(f"📝 Training summary saved to: {summary_path}")
        
        if use_wandb:
            wandb.finish()
            print("✅ Check full results at: https://wandb.ai/home")
        
        return best_accuracy
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to cleanup WandB
        try:
            if 'use_wandb' in locals() and use_wandb:
                wandb.finish()
        except:
            pass
        
        return None


# Entry point with multiprocessing safety
if __name__ == "__main__":
    # Windows requires freeze_support for multiprocessing
    if os.name == 'nt':
        multiprocessing.freeze_support()
    
    # Run training
    best_acc = main()
    
    if best_acc is not None:
        print(f"\n✨ Training finished successfully!")
        if best_acc > 0:
            print(f"   Best accuracy: {best_acc:.2f}%")
    else:
        print(f"\n❌ Training failed or was interrupted")