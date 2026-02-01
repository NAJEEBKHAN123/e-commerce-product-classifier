# /content/e-commerce-product-classifier/src/pipeline/model_training.py
import multiprocessing
import sys
import os

# ================= PROJECT PATH =================
# Get the absolute path to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
print(f"📁 Project root: {project_root}")
print(f"📂 Current directory: {os.getcwd()}")

# ================= MATPLOTLIB FIX =================
try:
    import matplotlib
    matplotlib.use("Agg")
    print("✅ Set matplotlib to Agg backend")
except ImportError:
    print("⚠️ matplotlib not available")

# ================= IMPORTS =================
import torch
import torch.nn as nn
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(project_root, '.env'))

# Try to import local modules
try:
    from src.data.loader import create_dataloaders
    from src.model.cnn import ProductCNN
    from src.model.train import Trainer
    from src.model.evaluation import Evaluator
    print("✅ All local modules imported successfully")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Trying alternative import method...")
    
    # Manual imports as fallback
    import importlib.util
    
    # Import cnn
    cnn_path = os.path.join(project_root, "src", "model", "cnn.py")
    if os.path.exists(cnn_path):
        spec = importlib.util.spec_from_file_location("cnn", cnn_path)
        cnn_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cnn_module)
        ProductCNN = cnn_module.ProductCNN
        print("✅ ProductCNN imported via direct path")
    
    # Import loader
    loader_path = os.path.join(project_root, "src", "data", "loader.py")
    if os.path.exists(loader_path):
        spec = importlib.util.spec_from_file_location("loader", loader_path)
        loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader_module)
        create_dataloaders = loader_module.create_dataloaders
        print("✅ create_dataloaders imported via direct path")

# Try to import WandB (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not available, running without logging")

def main():
    """Main training function"""
    
    # ========== TRAINING CONFIG ==========
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    PATIENCE = 25  # Increased from 5 to 25
    ACCURACY_TOLERANCE = 0.2  # Allow 0.2% fluctuation without penalty
    
    # Dataset path - adjust for your setup
    DATA_DIR =  "/content/drive/MyDrive/dataset"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========== PRINT CONFIG ==========
    print("=" * 60)
    print("🛒 E-COMMERCE PRODUCT CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"📱 Device: {DEVICE}")
    print(f"📊 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🎯 Learning rate: {LEARNING_RATE}")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"⏳ Early stopping patience: {PATIENCE}")
    print(f"📈 Accuracy tolerance: {ACCURACY_TOLERANCE}%")
    print("=" * 60)
    
    # ========== DATA CHECK ==========
    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset not found: {DATA_DIR}")
        print("💡 Please check if:")
        print("   1. Google Drive is mounted")
        print("   2. Dataset exists at the specified path")
        print("   3. .env file has correct DATASET_PATH")
        return None
    print("✅ Dataset found")
    
    # Check dataset structure
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    
    if not os.path.exists(train_dir):
        print(f"❌ Train directory not found: {train_dir}")
        return None
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return None
    
    print(f"✅ Train directory: {train_dir}")
    print(f"✅ Validation directory: {val_dir}")
    
    # ========== WANDB INITIALIZATION ==========
    use_wandb = False
    if WANDB_AVAILABLE:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            try:
                wandb.init(
                    project="ecommerce-product-classifier",
                    name=f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "device": DEVICE,
                        "model": "ProductCNN-ResNet50",
                        "patience": PATIENCE,
                        "accuracy_tolerance": ACCURACY_TOLERANCE
                    }
                )
                use_wandb = True
                print("✅ WandB initialized")
            except Exception as e:
                print(f"⚠️ WandB failed: {e}")
        else:
            print("ℹ️ WANDB_API_KEY not found, running without WandB")
    else:
        print("ℹ️ Running without WandB")
    
    # ========== DATALOADERS ==========
    print("\n📊 Creating dataloaders...")
    try:
        # Determine number of workers
        if os.name == "nt":  # Windows
            num_workers = 0
        else:  # Linux/Colab
            num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 2
        
        print(f"📊 Using num_workers={num_workers} for DataLoader")
        
        train_loader, val_loader, _, class_weights, categories = create_dataloaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=num_workers
        )
        
        print(f"✅ Categories ({len(categories)}): {categories}")
        print(f"✅ Train images: {len(train_loader.dataset)}")
        print(f"✅ Validation images: {len(val_loader.dataset)}")
        
        if class_weights is not None:
            class_weights = class_weights.to(DEVICE)
            print(f"✅ Class weights loaded for imbalance handling")
        
    except Exception as e:
        print(f"❌ Dataloader error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== MODEL INITIALIZATION ==========
    print("\n🤖 Initializing model...")
    try:
        model = ProductCNN(num_classes=len(categories)).to(DEVICE)
        print(f"✅ Model loaded on {DEVICE}")
        print(f"📊 Model parameters: {model.get_parameter_count():,}")
        
        # Print model architecture summary
        print("\n📋 Model Architecture:")
        print("-" * 40)
        print(model)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== TRAINER & EVALUATOR SETUP ==========
    print("\n⚙️ Setting up trainer and evaluator...")
    try:
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
        
        print("✅ Trainer and evaluator initialized")
        
    except Exception as e:
        print(f"❌ Trainer/evaluator setup error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== TRAINING LOOP ==========
    print("\n" + "🚀" * 20)
    print("🚀 STARTING TRAINING")
    print("🚀" * 20)
    
    best_accuracy = 0.0
    no_improve = 0
    training_history = []
    
    for epoch in range(EPOCHS):
        print("\n" + "=" * 50)
        print(f"📅 EPOCH {epoch + 1}/{EPOCHS}")
        print("=" * 50)
        
        # Training phase
        print("🎯 Training...")
        train_loss, train_acc = trainer.start_training_loop(epoch + 1)
        
        if train_loss is None:
            print("❌ Training failed this epoch")
            continue
        
        # Validation phase
        print("📊 Evaluating...")
        val_results = evaluator.start_evaluation_loop(epoch + 1)
        
        if val_results:
            val_loss = val_results["average_loss"]
            val_acc = val_results["accuracy"]
            
            # Store history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # Get current learning rate
            current_lr = trainer.current_lr if hasattr(trainer, 'current_lr') else LEARNING_RATE
            
            # Print metrics
            print("\n📈 EPOCH SUMMARY:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            print(f"   Best Accuracy: {best_accuracy:.2f}%")
            
            # Log to WandB
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr
                })
            
            # ========== IMPROVED SAVING LOGIC ==========
            # Save if: 1. New best OR 2. Within tolerance and LR just decreased
            save_model = False
            message = ""
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                save_model = True
                no_improve = 0
                message = f"💾 NEW BEST MODEL SAVED (Accuracy: {val_acc:.2f}%)"
            elif val_acc >= best_accuracy - ACCURACY_TOLERANCE:
                # Within tolerance - check if learning rate just decreased
                if epoch > 0 and hasattr(trainer, 'previous_lr'):
                    if trainer.previous_lr > current_lr:
                        save_model = True
                        message = f"💾 Saved after LR decrease: {trainer.previous_lr:.6f} → {current_lr:.6f}"
                        no_improve = max(0, no_improve - 3)  # Partial reset
                    else:
                        no_improve += 1
                        message = f"📊 Maintaining good accuracy: {val_acc:.2f}% (Best: {best_accuracy:.2f}%)"
                else:
                    no_improve += 1
                    message = f"📊 Within tolerance: {val_acc:.2f}% (Best: {best_accuracy:.2f}%)"
            else:
                no_improve += 1
                message = f"⏳ No improvement: {no_improve}/{PATIENCE}"
            
            print(message)
            
            # Save model if conditions met
            if save_model:
                trainer.save_model(epoch + 1, val_acc)
            
            # Store previous LR for comparison
            if hasattr(trainer, 'current_lr'):
                trainer.previous_lr = current_lr
            
            # ========== SMART EARLY STOPPING ==========
            if no_improve >= PATIENCE:
                print(f"🛑 Early stopping check triggered at epoch {epoch + 1}")
                
                # Check if we should give model one more chance with LR adjustment
                if epoch < 30:  # Only for first 30 epochs
                    print("🔄 Attempting one final LR adjustment before stopping...")
                    if hasattr(trainer, 'scheduler'):
                        old_lr = current_lr
                        # Force a learning rate reduction
                        trainer.scheduler.step(val_loss)
                        new_lr = trainer.optimizer.param_groups[0]['lr']
                        
                        if new_lr < old_lr:
                            print(f"✅ LR reduced from {old_lr:.6f} to {new_lr:.6f}")
                            print(f"🔄 Continuing training for {PATIENCE//2} more epochs...")
                            no_improve = max(0, no_improve - PATIENCE//2)  # Half reset
                        else:
                            print("❌ LR already at minimum, stopping training")
                            break
                    else:
                        print("❌ No scheduler available, stopping training")
                        break
                else:
                    print("✅ Model had sufficient training epochs, stopping")
                    break
        else:
            print("❌ Validation failed this epoch")
            no_improve += 1  # Count validation failure as no improvement
    
    # ========== TRAINING COMPLETE ==========
    print("\n" + "🎉" * 20)
    print("🎉 TRAINING COMPLETE")
    print("🎉" * 20)
    
    # Save final model
    try:
        final_model_path = os.path.join("models", f"ecommerce_cnn_final_acc{best_accuracy:.2f}.pth")
        trainer.save_model(epoch + 1, best_accuracy)
        print(f"💾 Final model saved with accuracy: {best_accuracy:.2f}%")
    except Exception as e:
        print(f"⚠️ Could not save final model: {e}")
    
    # Print comprehensive summary
    print("\n📊 TRAINING SUMMARY:")
    print("-" * 40)
    print(f"Total epochs trained: {len(training_history)}")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    if training_history:
        print(f"Final training accuracy: {training_history[-1]['train_acc']:.2f}%")
        print(f"Final validation accuracy: {training_history[-1]['val_acc']:.2f}%")
        print(f"Training/Validation gap: {training_history[-1]['train_acc'] - training_history[-1]['val_acc']:.2f}%")
        
        # Calculate improvements
        if len(training_history) > 1:
            first_acc = training_history[0]['val_acc']
            improvement = best_accuracy - first_acc
            print(f"Total improvement: {improvement:.2f}%")
    
    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        epochs = [h['epoch'] for h in training_history]
        train_losses = [h['train_loss'] for h in training_history]
        val_losses = [h['val_loss'] for h in training_history]
        train_accs = [h['train_acc'] for h in training_history]
        val_accs = [h['val_acc'] for h in training_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
        ax2.axhline(y=best_accuracy, color='g', linestyle='--', label=f'Best: {best_accuracy:.2f}%', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print(f"📈 Training history saved to: training_history.png")
        
    except ImportError:
        print("⚠️ Matplotlib not available, skipping plots")
    
    # Cleanup
    if use_wandb:
        wandb.finish()
        print("✅ WandB session ended")
    
    return best_accuracy


if __name__ == "__main__":
    # Windows specific setup
    if os.name == "nt":
        multiprocessing.freeze_support()
    
    try:
        print("\n" + "🌟" * 30)
        print("🌟 E-commerce Product Classifier Training")
        print("🌟" * 30)
        
        best_acc = main()
        
        if best_acc is not None:
            print(f"\n✨ TRAINING FINISHED SUCCESSFULLY!")
            print(f"✨ Best accuracy achieved: {best_acc:.2f}%")
        else:
            print("\n❌ Training failed to complete")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("   1. Check dataset path and structure")
        print("   2. Verify all required files exist")
        print("   3. Check .env file for correct paths")
        print("   4. Ensure all dependencies are installed")