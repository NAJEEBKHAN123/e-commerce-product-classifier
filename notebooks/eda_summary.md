# E-commerce Product Image Dataset - EDA Summary

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

## Class Imbalance
- Imbalance Ratio: 3.71x
- Most common: GROCERY (3,978 images)
- Least common: CLOTHING_ACCESSORIES_JEWELLERY (1,071 images)

## Recommendations
1. Resize images to consistent dimensions
2. Apply data augmentation techniques
3. Use class weights in loss function
4. Monitor validation performance per category

*Generated on 2026-01-29 08:58:54*
