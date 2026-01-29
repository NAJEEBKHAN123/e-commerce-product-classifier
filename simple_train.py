# simple_train.py (place in project root)
import sys
import os

# ✅ Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*60)
print("SIMPLE TRAINING TEST")
print("="*60)
print(f"Project root: {project_root}")

try:
    # Test imports
    from src.model.cnn import ProductCNN
    from src.model.train import Trainer
    from src.model.evaluation import Evaluator
    print("✅ All imports successful!")
    
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    # Test model creation
    model = ProductCNN(num_classes=9)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\n" + "="*60)
    print("TEST PASSED! Your structure is correct.")
    print("="*60)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
    print("\nDebugging...")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory:")
    for item in os.listdir('.'):
        print(f"  {item}")
    
    if os.path.exists('src'):
        print(f"\nContents of src folder:")
        for item in os.listdir('src'):
            print(f"  {item}")
        
        if os.path.exists('src/model'):
            print(f"\nContents of src/model folder:")
            for item in os.listdir('src/model'):
                print(f"  {item}")
    
    print("\n" + "="*60)
    print("FIX: Make sure you have:")
    print("1. __init__.py files in src/, src/model/, etc.")
    print("2. Run from project root directory")
    print("3. Files exist: src/model/cnn.py, src/model/train.py")
    print("="*60)