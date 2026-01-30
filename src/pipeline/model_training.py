import multiprocessing
import sys
import os

# ================= PROJECT PATH =================
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.insert(0, project_root)
print(f"📁 Project root: {project_root}")

# ================= MATPLOTLIB FIX =================
try:
    import matplotlib
    matplotlib.use("Agg")
    print("✅ Set matplotlib to Agg backend")
except ImportError:
    print("⚠️ matplotlib not available")

# ================= IMPORTS =================
import torch
from src.data.loader import create_dataloaders
from src.model.cnn import ProductCNN
from src.model.train import Trainer
from src.model.evaluation import Evaluator
import wandb
from datetime import datetime
from dotenv import load_dotenv

# ================= ENV VARIABLES =================
load_dotenv()

# ================= AMP / Mixed Precision =================
use_amp = torch.cuda.is_available()
if use_amp:
    print("⚡ Mixed precision (AMP) enabled for faster GPU training")

def main():
    """Main training function"""

    # ========== TRAINING CONFIG ==========
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    PATIENCE = 5

    DATA_DIR = os.getenv(
        "DATASET_PATH",
        "/content/drive/MyDrive/dataset"
    )

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
    print("=" * 60)

    # ========== DATA CHECK ==========
    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset not found: {DATA_DIR}")
        return None
    print("✅ Dataset found")

    # ========== WANDB ==========
    wandb_api_key = os.getenv("WANDB_API_KEY")
    use_wandb = False

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
                    "model": "ProductCNN"
                }
            )
            use_wandb = True
            print("✅ WandB initialized")
        except Exception as e:
            print(f"⚠️ WandB failed: {e}")
    else:
        print("ℹ️ Running without WandB")

    # ========== DATALOADERS ==========
    print("\n📊 Creating dataloaders...")
    try:
        if os.name == "nt":
            num_workers = 0
        else:
            cpu_count = multiprocessing.cpu_count()
            num_workers = min(8, cpu_count // 2)  # faster for Colab/Linux

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

    except Exception as e:
        print(f"❌ Dataloader error: {e}")
        return None

    # ========== MODEL ==========
    print("\n🤖 Initializing model...")
    model = ProductCNN(num_classes=len(categories)).to(DEVICE)
    print(f"✅ Model loaded on {DEVICE}")

    # ========== TRAINER / EVALUATOR ==========
    trainer = Trainer(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        data_loader=train_loader,
        model=model,
        model_path="models/ecommerce_cnn",
        device=DEVICE,
        class_weights=class_weights,
        use_amp=use_amp  # <- AMP flag for trainer
    )

    evaluator = Evaluator(
        batch_size=BATCH_SIZE,
        data_loader=val_loader,
        model=model,
        device=DEVICE
    )

    best_accuracy = 0.0
    no_improve = 0

    # ========== TRAINING LOOP ==========
    print("\n🚀 STARTING TRAINING")
    for epoch in range(EPOCHS):
        print("\n" + "=" * 40)
        print(f"📅 Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 40)

        # Training
        train_loss, train_acc = trainer.start_training_loop(epoch + 1)

        if train_loss is None:
            print("❌ Training failed")
            break

        # Validation
        val_results = evaluator.start_evaluation_loop(epoch + 1)
        if val_results:
            val_loss = val_results["average_loss"]
            val_acc = val_results["accuracy"]

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}%"
            )
            print(
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                })

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                trainer.save_model(epoch + 1, val_acc)
                no_improve = 0
                print("💾 New best model saved")
            else:
                no_improve += 1
                print(f"⏳ No improvement: {no_improve}/{PATIENCE}")
                if no_improve >= PATIENCE:
                    print("🛑 Early stopping")
                    break

    print("\n🎉 TRAINING COMPLETE")
    print(f"🏆 Best accuracy: {best_accuracy:.2f}%")

    trainer.save_model(EPOCHS, best_accuracy, final=True)
    if use_wandb:
        wandb.finish()

    return best_accuracy


if __name__ == "__main__":
    if os.name == "nt":
        multiprocessing.freeze_support()

    try:
        best_acc = main()
        if best_acc is not None:
            print(f"\n✨ Finished! Best accuracy: {best_acc:.2f}%")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
