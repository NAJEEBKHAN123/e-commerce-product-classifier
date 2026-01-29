# E-commerce Product Image Dataset - EDA Report

## Dataset Overview
- Total Images: 18,175
- Categories: 9
- Splits: 3

## Split Distribution
- train: 13,992 images (77.0%)
- val: 3,632 images (20.0%)
- check: 551 images (3.0%)

## Top 5 Categories
1. GROCERY: 5,166 images (28.4%)
2. HOME_KITCHEN_TOOLS: 2,228 images (12.3%)
3. ELECTRONICS: 1,757 images (9.7%)
4. PET_SUPPLIES: 1,637 images (9.0%)
5. SPORTS_OUTDOOR: 1,605 images (8.8%)

## Image Characteristics (Sample Analysis)
- Average Resolution: 224 × 224 pixels
- Average File Size: 7.8 KB
- Average Brightness: 198.1
- Average Contrast: 64.9
- Portrait Images: 0.0%

## Class Imbalance
- Imbalance Ratio: 3.71x
- Most common category: GROCERY (3,978 images)
- Least common category: CLOTHING_ACCESSORIES_JEWELLERY (1,071 images)

## Recommendations
1. Resize images to consistent dimensions (e.g., 224x224 or 299x299)
2. Apply data augmentation techniques
3. Use class weights in loss function
4. Consider brightness/contrast normalization
5. Monitor validation performance per category

*Report generated on 2026-01-29 08:53:17*
