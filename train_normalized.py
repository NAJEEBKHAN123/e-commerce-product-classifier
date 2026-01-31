# train_normalized.py
import sys
import os

# Use normalized dataset
os.environ['DATASET_PATH'] = 'dataset_normalized'

# Import and run your training
from src.pipeline.model_training import main

print("🚀 TRAINING WITH NORMALIZED DATASET")
print("="*60)
best_acc = main()