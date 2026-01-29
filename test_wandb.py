# test_wandb.py
import os
import wandb

print("Testing WandB Setup...")
print("="*50)

# Check if API key is set
if "WANDB_API_KEY" in os.environ:
    print("✅ API Key found in environment")
    print(f"Key ID: {os.environ['WANDB_API_KEY'][:30]}...")
else:
    print("❌ API Key NOT found in environment")
    print("Set it with: set WANDB_API_KEY=your_key (Windows)")
    print("Or: export WANDB_API_KEY=your_key (Mac/Linux)")
    exit(1)

# Test connection
try:
    # Initialize in dry run mode first
    wandb.init(project="test-connection", mode="dryrun")
    print("✅ WandB initialized successfully")
    wandb.finish()
    
    # Now test real connection
    print("\nTesting real connection to WandB servers...")
    wandb.init(project="ecommerce-product-classifier", 
               name="test-run",
               config={"test": True})
    print("✅ Successfully connected to WandB!")
    print("✅ You can now track your experiments")
    wandb.finish()
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("Possible issues:")
    print("1. Internet connection")
    print("2. Firewall blocking WandB")
    print("3. Invalid API key")