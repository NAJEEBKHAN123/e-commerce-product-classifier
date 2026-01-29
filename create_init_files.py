# create_init_files.py
import os

print("Creating __init__.py files...")

# Directories that need __init__.py
directories = [
    "src",
    "src/data",
    "src/model", 
    "src/pipeline"
]

for directory in directories:
    init_file = os.path.join(directory, "__init__.py")
    
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)
    
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# Package initialization file\n")
            f.write(f'"""\n{os.path.basename(directory)} module\n"""\n')
        print(f"✅ Created: {init_file}")
    else:
        print(f"✓ Already exists: {init_file}")

print("\nChecking project structure...")

# List all Python files
print("\nPython files in project:")
for root, dirs, files in os.walk("."):
    # Skip virtual environment
    if ".venv" in root or "__pycache__" in root:
        continue
    
    for file in files:
        if file.endswith(".py"):
            rel_path = os.path.relpath(os.path.join(root, file))
            print(f"  📄 {rel_path}")

print("\n✅ Setup complete!")