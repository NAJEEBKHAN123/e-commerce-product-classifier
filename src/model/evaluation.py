# src/model/evaluation.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    """Evaluation class for model validation/testing"""
    
    def __init__(self, batch_size: int, data_loader: DataLoader, model, device: str):
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Your 9 categories
        self.categories = [
            'BABY_PRODUCTS',
            'BEAUTY_HEALTH', 
            'CLOTHING_ACCESSORIES_JEWELLERY',
            'ELECTRONICS',
            'GROCERY',
            'HOBBY_ARTS_STATIONERY',
            'HOME_KITCHEN_TOOLS',
            'PET_SUPPLIES',
            'SPORTS_OUTDOOR'
        ]
    
    def start_evaluation_loop(self, epoch=None):
        """Run evaluation on validation/test set"""
        try:
            self.model.eval()  # Set model to evaluation mode
            total_loss = 0
            correct = 0
            total = 0
            
            # Store predictions for confusion matrix
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            with torch.no_grad():  # Disable gradient computation
                for batch_idx, (images, labels, _) in enumerate(self.data_loader):
                    # Move data to device
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    
                    # Calculate metrics
                    total_loss += loss.item()
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Store for detailed analysis
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
                    # Print progress
                    if batch_idx % 5 == 0:
                        if epoch is not None:
                            print(f"Validation -> Epoch {epoch} -> Batch {batch_idx}: Loss = {loss.item():.4f}")
                        else:
                            print(f"Evaluation -> Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            # Calculate final metrics
            avg_loss = total_loss / len(self.data_loader)
            accuracy = 100.0 * correct / total
            
            # Print summary
            if epoch is not None:
                print(f"Validation -> Epoch {epoch}:")
            else:
                print("Evaluation Results:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Correct: {correct}/{total}")
            
            # Convert to numpy arrays for sklearn
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            return {
                'average_loss': avg_loss,
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'predictions': all_predictions,
                'labels': all_labels,
                'probabilities': np.array(all_probabilities)
            }
            
        except Exception as e:
            print(f"Error in evaluation loop: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_confusion_matrix(self, predictions, labels, save_path=None):
        """Generate and display confusion matrix"""
        try:
            cm = confusion_matrix(labels, predictions)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.categories,
                       yticklabels=self.categories)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Confusion matrix saved to: {save_path}")
            
            plt.show()
            
            return cm
            
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            return None
    
    def generate_classification_report(self, predictions, labels):
        """Generate detailed classification report"""
        try:
            report = classification_report(
                labels, 
                predictions, 
                target_names=self.categories,
                output_dict=True
            )
            
            print("\n" + "="*60)
            print("CLASSIFICATION REPORT")
            print("="*60)
            
            # Print per-class metrics
            print(f"{'Class':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-"*70)
            
            for idx, category in enumerate(self.categories):
                if str(idx) in report:
                    metrics = report[str(idx)]
                    print(f"{category:<30} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                          f"{metrics['f1-score']:<10.3f} {int(metrics['support']):<10}")
            
            # Print overall metrics
            print("-"*70)
            overall = report['weighted avg']
            print(f"{'WEIGHTED AVERAGE':<30} {overall['precision']:<10.3f} {overall['recall']:<10.3f} "
                  f"{overall['f1-score']:<10.3f} {int(report['macro avg']['support']):<10}")
            print(f"{'ACCURACY':<30} {'':<10} {'':<10} {report['accuracy']:<10.3f} {int(sum(report[str(i)]['support'] for i in range(len(self.categories)))):<10}")
            
            return report
            
        except Exception as e:
            print(f"Error generating classification report: {e}")
            return None
    
    def analyze_misclassifications(self, data_loader, predictions, labels, top_n=10):
        """Analyze top misclassified images"""
        try:
            misclassified_indices = np.where(predictions != labels)[0]
            
            if len(misclassified_indices) == 0:
                print("No misclassifications found!")
                return
            
            print(f"\nAnalyzing {len(misclassified_indices)} misclassifications...")
            
            # Get image paths for misclassified items
            all_image_paths = []
            for _, _, paths in data_loader:
                all_image_paths.extend(paths)
            
            # Display top N misclassifications
            print(f"\nTop {min(top_n, len(misclassified_indices))} Misclassifications:")
            print("="*80)
            print(f"{'Image':<30} {'True':<20} {'Predicted':<20} {'Confidence':<10}")
            print("-"*80)
            
            for i, idx in enumerate(misclassified_indices[:top_n]):
                if idx < len(all_image_paths):
                    true_label = self.categories[labels[idx]]
                    pred_label = self.categories[predictions[idx]]
                    img_path = all_image_paths[idx]
                    
                    # Get just the filename
                    img_name = img_path.split('/')[-1] if '/' in img_path else img_path.split('\\')[-1]
                    
                    print(f"{img_name:<30} {true_label:<20} {pred_label:<20}")
            
        except Exception as e:
            print(f"Error analyzing misclassifications: {e}")
    
    def evaluate_model(self, save_results=False):
        """Complete model evaluation with all metrics"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Run evaluation
        results = self.start_evaluation_loop()
        
        if results is None:
            print("Evaluation failed!")
            return None
        
        # Generate confusion matrix
        cm_path = "confusion_matrix.png" if save_results else None
        cm = self.generate_confusion_matrix(
            results['predictions'], 
            results['labels'],
            save_path=cm_path
        )
        
        # Generate classification report
        report = self.generate_classification_report(
            results['predictions'], 
            results['labels']
        )
        
        # Analyze misclassifications
        self.analyze_misclassifications(
            self.data_loader,
            results['predictions'],
            results['labels'],
            top_n=10
        )
        
        # Save results if requested
        if save_results:
            import json
            import pandas as pd
            
            # Save metrics
            metrics = {
                'accuracy': results['accuracy'],
                'average_loss': results['average_loss'],
                'correct': results['correct'],
                'total': results['total']
            }
            
            with open('evaluation_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'true_label': [self.categories[i] for i in results['labels']],
                'predicted_label': [self.categories[i] for i in results['predictions']],
                'is_correct': results['predictions'] == results['labels']
            })
            predictions_df.to_csv('predictions.csv', index=False)
            
            print(f"\nEvaluation results saved to:")
            print(f"  - evaluation_metrics.json")
            print(f"  - predictions.csv")
            if cm_path:
                print(f"  - {cm_path}")
        
        return {
            'metrics': results,
            'confusion_matrix': cm,
            'classification_report': report
        }