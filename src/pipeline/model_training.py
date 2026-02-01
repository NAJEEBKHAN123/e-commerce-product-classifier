import multiprocessing
import sys
import os

# ================= PROJECT PATH =================
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

load_dotenv(os.path.join(project_root, ".env"))

# ================= LOCAL MODULE IMPORTS =================
try:
    from src.data.loader import create_dataloaders
    from src.model.cnn import ProductCNN
    from src.model.train import Trainer
    from src.model.evaluation import Evaluator
    print("✅ All local modules imported successfully")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")

# ================= WANDB (OPTIONAL) =================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not available")

# ======================================================
# ================= MAIN FUNCTION ======================
# ======================================================
def main():
    """Main training function"""
    
    # ========== TRAINING CONFIG ==========
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    PATIENCE = 25
    ACCURACY_TOLERANCE = 0.2
    
    # Dataset path
    DATA_DIR = os.getenv(
        "DATASET_PATH",
        "/content/drive/MyDrive/dataset"
    )
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    # ===== PRINT CONFIG =====
    print("=" * 60)
    print("🛒 E-COMMERCE PRODUCT CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"📱 Device: {DEVICE}")
    print(f"📊 Epochs: {EPOCHS}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🎯 Learning rate: {LEARNING_RATE}")
    print(f"📁 Data directory: {DATA_DIR}")
    print("=" * 60)

    # ===== DATA CHECK =====
    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset not found: {DATA_DIR}")
        return None

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ Train/Val folders missing")
        return None

    print("✅ Dataset structure OK")

    # ===== DATALOADERS =====
    num_workers = 0 if os.name == "nt" else 2

    train_loader, val_loader, _, class_weights, categories = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=num_workers
    )

    if class_weights is not None:
        class_weights = class_weights.to(DEVICE)

    print(f"✅ Categories: {categories}")

    # ===== MODEL =====
    model = ProductCNN(num_classes=len(categories)).to(DEVICE)
    print(f"✅ Model loaded on {DEVICE}")

    # ===== TRAINER =====
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

    # ===== TRAINING LOOP =====
    best_accuracy = 0
    no_improve = 0
    history = []

    for epoch in range(EPOCHS):

        print(f"\n📅 Epoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = trainer.start_training_loop(epoch+1)

        val_results = evaluator.start_evaluation_loop(epoch+1)

        if not val_results:
            continue

        val_loss = val_results["average_loss"]
        val_acc = val_results["accuracy"]

        history.append((train_loss, val_loss, train_acc, val_acc))

        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        # ===== SAVE BEST =====
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            no_improve = 0
            trainer.save_model(epoch+1, val_acc)
            print("💾 New best model saved")
        else:
            no_improve += 1

        # ===== EARLY STOP =====
        if no_improve >= PATIENCE:
            print("🛑 Early stopping")
            break

    print("\n🎉 Training Complete")
    print(f"Best Accuracy: {best_accuracy:.2f}%")

    return best_accuracy


# ======================================================
# ================= RUN SCRIPT =========================
# ======================================================
if __name__ == "__main__":

    if os.name == "nt":
        multiprocessing.freeze_support()

    try:
        best = main()

        if best:
            print(f"✨ Finished! Best accuracy: {best:.2f}%")
        else:
            print("❌ Training failed")

    except KeyboardInterrupt:
        print("⚠️ Interrupted")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
