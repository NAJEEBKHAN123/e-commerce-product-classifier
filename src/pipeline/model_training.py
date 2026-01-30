# src/pipeline/model_training.py - UPDATED FOR COLAB
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")

# Try to set matplotlib backend to avoid issues
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    print("✅ Set matplotlib to Agg backend")
except:
    print("⚠️  Could not set matplotlib backend")

import torch
from src.data.loader import create_dataloaders
from src.model.cnn import ProductCNN
from src.model.train import Trainer
from src.model.evaluation import Evaluator
import wandb
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

def main():
    try:
        # Training Config
        EPOCHS = 10  # Changed from 2 to 10 for proper training
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        
        # FIXED: Use Google Drive path for Colab, with Windows fallback for VS Code
        DATA_DIR = os.getenv("DATASET_PATH", "/content/drive/MyDrive/dataset")
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("="*60)
        print("E-COMMERCE PRODUCT CLASSIFIER TRAINING")
        print("="*60)
        print(f"Device: {DEVICE}")
        print(f"Epochs: {EPOCHS}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Data directory: {DATA_DIR}")
        
        # Check dataset
        if not os.path.exists(DATA_DIR):
            print(f"❌ Dataset not found: {DATA_DIR}")
            print("Trying alternative paths...")
            
            # Try Windows path (for VS Code)
            windows_path = "d:\\ecommerce-product-classifier\\dataset"
            if os.path.exists(windows_path):
                DATA_DIR = windows_path
                print(f"✅ Found dataset at Windows path: {DATA_DIR}")
            else:
                # Try current directory
                current_path = "dataset"
                if os.path.exists(current_path):
                    DATA_DIR = current_path
                    print(f"✅ Found dataset at: {DATA_DIR}")
                else:
                    print("❌ Dataset not found in any location")
                    return
        
        print("✅ Dataset found")
        
        # Initialize WandB (optional)
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            try:
                wandb.init(
                    project="ecommerce-product-classifier",
                    config={
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "device": DEVICE
                    },
                    name=f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                print("✅ WandB initialized")
            except Exception as e:
                print(f"⚠️  WandB initialization failed: {e}")
                wandb_api_key = None
        else:
            print("⚠️  Running without WandB")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        try:
            train_loader, val_loader, _, class_weights, categories = create_dataloaders(
                data_dir=DATA_DIR,
                batch_size=BATCH_SIZE,
                num_workers=2  # Changed to 2 for Colab (0 was for Windows)
            )
            print(f"✅ Categories: {categories}")
            print(f"✅ Train batches: {len(train_loader)}")
            print(f"✅ Val batches: {len(val_loader)}")
            if class_weights is not None:
                print(f"✅ Using class weights for imbalance")
        except Exception as e:
            print(f"❌ Error creating dataloaders: {e}")
            print("\nDebugging dataset structure...")
            
            # List dataset contents
            if os.path.exists(DATA_DIR):
                for split in ['train', 'val', 'check']:
                    split_path = os.path.join(DATA_DIR, split)
                    if os.path.exists(split_path):
                        print(f"\n{split.upper()} folder:")
                        categories = os.listdir(split_path)
                        print(f"  Categories: {len(categories)}")
                        for cat in categories[:3]:  # Show first 3
                            cat_path = os.path.join(split_path, cat)
                            if os.path.isdir(cat_path):
                                images = [f for f in os.listdir(cat_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                print(f"  {cat}: {len(images)} images")
            return
        
        # Initialize model
        print("\nInitializing model...")
        try:
            model = ProductCNN(num_classes=len(categories)).to(DEVICE)
            print(f"✅ Model initialized on {DEVICE}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            # Try with default 9 classes
            model = ProductCNN(num_classes=9).to(DEVICE)
            print(f"✅ Model initialized with 9 classes on {DEVICE}")
        
        # Initialize trainer and evaluator
        trainer = Trainer(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            data_loader=train_loader,
            model=model,
            model_path="ecommerce_cnn",
            device=DEVICE,
            class_weights=class_weights
        )
        
        evaluator = Evaluator(
            batch_size=BATCH_SIZE,
            data_loader=val_loader,
            model=model,
            device=DEVICE
        )
        
        best_accuracy = 0
        
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING - {EPOCHS} EPOCHS")
        print(f"{'='*60}")
        
        # Training loop
        for epoch in range(EPOCHS):
            print(f"\n{'='*40}")
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"{'='*40}")
            
            # Training
            train_loss, train_acc = trainer.start_training_loop(epoch+1)
            if train_loss is None:
                print("❌ Training failed, stopping...")
                break
            
            # Validation
            print("\nValidation...")
            val_results = evaluator.start_evaluation_loop(epoch+1)
            
            if val_results:
                val_loss = val_results['average_loss']
                val_acc = val_results['accuracy']
                
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Log to WandB
                if wandb_api_key:
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
                    model_path = trainer.save_model(epoch=epoch+1, accuracy=val_acc)
                    print(f"✅ New best model saved: {val_acc:.2f}%")
                    
                    if wandb_api_key and model_path:
                        wandb.save(model_path)
            else:
                print("❌ Validation failed")
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {best_accuracy:.2f}%")
        
        # Save final model
        final_path = trainer.save_model(epoch=EPOCHS, accuracy=best_accuracy)
        if final_path:
            print(f"Final model saved to: {final_path}")
        
        if wandb_api_key:
            wandb.finish()
            print("✅ Check results at: https://wandb.ai/home")
        
        return best_accuracy
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            if wandb_api_key:
                wandb.finish()
        except:
            pass
        
        return None

if __name__ == "__main__": 
    main()