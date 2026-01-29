#!/usr/bin/env python3
"""
Script to check dataset structure and statistics
"""

import os
import json
from PIL import Image
import matplotlib.pyplot as plt

def check_dataset_structure(dataset_path="dataset"):
    """Check the dataset structure and print statistics"""
    
    print("=" * 60)
    print("DATASET CHECKER")
    print("=" * 60)
    
    folders = ["train", "val", "check"]
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found!")
        return
    
    all_categories = set()
    statistics = {}
    
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: '{folder}' folder not found!")
            continue
            
        categories = sorted([d for d in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, d))])
        
        all_categories.update(categories)
        
        stats = {
            'total_images': 0,
            'categories': {},
            'image_sizes': []
        }
        
        print(f"\n{folder.upper()} SET:")
        print("-" * 40)
        
        for category in categories:
            cat_path = os.path.join(folder_path, category)
            
            # Count images
            images = [f for f in os.listdir(cat_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # Get sample image size
            sample_size = None
            if images:
                sample_img = os.path.join(cat_path, images[0])
                try:
                    with Image.open(sample_img) as img:
                        sample_size = img.size
                except:
                    sample_size = (0, 0)
            
            stats['categories'][category] = {
                'count': len(images),
                'sample_size': sample_size
            }
            stats['total_images'] += len(images)
            
            print(f"  {category}: {len(images):>4} images")
            
            # Collect image sizes from first 10 images
            for img_file in images[:10]:
                img_path = os.path.join(cat_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        stats['image_sizes'].append(img.size)
                except:
                    pass
        
        print(f"\n  Total: {stats['total_images']} images")
        statistics[folder] = stats
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    categories_list = sorted(list(all_categories))
    print(f"\nTotal Categories: {len(categories_list)}")
    print("Categories:", ", ".join(categories_list))
    
    # Create visualization
    create_visualization(statistics, categories_list)
    
    # Save statistics
    with open('dataset_statistics.json', 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\nStatistics saved to: dataset_statistics.json")
    
    return categories_list

def create_visualization(statistics, categories):
    """Create visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar chart: Images per set
    axes[0, 0].bar(statistics.keys(), 
                   [stats['total_images'] for stats in statistics.values()])
    axes[0, 0].set_title('Total Images per Dataset Split')
    axes[0, 0].set_ylabel('Number of Images')
    for i, v in enumerate([stats['total_images'] for stats in statistics.values()]):
        axes[0, 0].text(i, v + 10, str(v), ha='center')
    
    # 2. Stacked bar: Images per category in train set
    if 'train' in statistics:
        train_cats = statistics['train']['categories']
        categories = list(train_cats.keys())
        counts = [train_cats[cat]['count'] for cat in categories]
        
        axes[0, 1].bar(categories, counts)
        axes[0, 1].set_title('Images per Category (Train Set)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Image size distribution
    all_sizes = []
    for folder_stats in statistics.values():
        all_sizes.extend(folder_stats['image_sizes'])
    
    if all_sizes:
        widths = [w for w, h in all_sizes]
        heights = [h for w, h in all_sizes]
        
        axes[1, 0].scatter(widths, heights, alpha=0.5)
        axes[1, 0].set_title('Image Size Distribution')
        axes[1, 0].set_xlabel('Width (pixels)')
        axes[1, 0].set_ylabel('Height (pixels)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Pie chart: Dataset split
    total_images = sum([stats['total_images'] for stats in statistics.values()])
    if total_images > 0:
        sizes = [stats['total_images'] for stats in statistics.values()]
        labels = [f'{k} ({v/total_images:.1%})' 
                 for k, v in zip(statistics.keys(), sizes)]
        
        axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%')
        axes[1, 1].set_title('Dataset Split Distribution')
    
    plt.suptitle('Dataset Analysis - E-commerce Product Images', fontsize=16)
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to: dataset_analysis.png")

if __name__ == "__main__":
    check_dataset_structure()