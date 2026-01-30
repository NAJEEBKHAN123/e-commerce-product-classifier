# src/model/evaluation.py - UPDATED FOR COLAB
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

class Evaluator:
    """Evaluation class for model validation/testing"""
    
    def __init__(self, batch_size: int, data_loader: DataLoader, model, device: str):
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.model = model
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Try to get categories from data loader
        self.categories = self._get_categories()
    
    def _get_categories(self):
        """Extract categories from data loader or use defaults"""
        try:
            # Try to get dataset from data loader
            if hasattr(self.data_loader, 'dataset') and hasattr(self.data_loader.dataset, 'classes'):
                return self.data_loader.dataset.classes
            
            # Try to get from dataset attribute
            dataset = getattr(self.data_loader, 'dataset', None)
            if dataset and hasattr(dataset, 'classes'):
                return dataset.classes
            
            # Default categories for e-commerce products
            return [
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
        except:
            return [f'Class_{i}' for i in range(9)]
    
    def start_evaluation_loop(self, epoch=None):
        """Run evaluation on validation/test set"""
        try:
            self.model.eval()  # Set model to evaluation mode
            total_loss = 0
            correct = 0
            total = 0
            
            # Store predictions for detailed analysis
            all_predictions = []
            all_labels = []
            all_probabilities = []
            all_confidences = []
            
            with torch.no_grad():  # Disable gradient computation
                for batch_idx, batch_data in enumerate(self.data_loader):
                    # Handle different data loader formats
                    if len(batch_data) == 3:
                        images, labels, _ = batch_data
                    elif len(batch_data) == 2:
                        images, labels = batch_data
                    else:
                        print(f"❌ Unexpected batch format: {len(batch_data)} elements")
                        continue
                    
                    # Move data to device
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    
                    # Calculate metrics
                    total_loss += loss.item()
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidences, predicted = torch.max(probabilities, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Store for detailed analysis
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_confidences.extend(confidences.cpu().numpy())
                    
                    # Print progress
                    if batch_idx % 5 == 0 and epoch is not None:
                        print(f"Validation -> Epoch {epoch} -> Batch {batch_idx}: "
                              f"Loss = {loss.item():.4f}, Acc = {100.0 * (predicted == labels).sum().item() / labels.size(0):.2f}%")
                    elif batch_idx % 5 == 0:
                        print(f"Evaluation -> Batch {batch_idx}: "
                              f"Loss = {loss.item():.4f}, Acc = {100.0 * (predicted == labels).sum().item() / labels.size(0):.2f}%")
            
            # Calculate final metrics
            avg_loss = total_loss / len(self.data_loader)
            accuracy = 100.0 * correct / total
            
            # Calculate additional metrics
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            all_confidences = np.array(all_confidences)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
            
            # Print summary
            if epoch is not None:
                print(f"\nValidation -> Epoch {epoch}:")
            else:
                print("\nEvaluation Results:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Avg Confidence: {np.mean(all_confidences):.4f}")
            
            return {
                'average_loss': avg_loss,
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_confidence': np.mean(all_confidences),
                'predictions': all_predictions,
                'labels': all_labels,
                'probabilities': np.array(all_probabilities),
                'confidences': all_confidences
            }
            
        except Exception as e:
            print(f"Error in evaluation loop: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_confusion_matrix(self, predictions, labels, save_path=None, show_plot=True):
        """Generate and display confusion matrix"""
        try:
            cm = confusion_matrix(labels, predictions)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.categories,
                       yticklabels=self.categories,
                       cbar_kws={'label': 'Count'})
            plt.title('Confusion Matrix', fontsize=16, pad=20)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Confusion matrix saved to: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return cm
            
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            return None
    
    def generate_classification_report(self, predictions, labels, save_path=None):
        """Generate detailed classification report"""
        try:
            report = classification_report(
                labels, 
                predictions, 
                target_names=self.categories,
                output_dict=True
            )
            
            # Print to console
            print("\n" + "="*70)
            print("CLASSIFICATION REPORT")
            print("="*70)
            
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
            
            accuracy = report['accuracy']
            total_samples = sum(report[str(i)]['support'] for i in range(len(self.categories)))
            print(f"{'ACCURACY':<30} {'':<10} {'':<10} {accuracy:<10.3f} {int(total_samples):<10}")
            print("="*70)
            
            # Save to file if requested
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save as JSON
                import json
                with open(save_path.replace('.txt', '.json'), 'w') as f:
                    json.dump(report, f, indent=2)
                
                # Save as readable text
                with open(save_path, 'w') as f:
                    f.write(classification_report(labels, predictions, target_names=self.categories))
                
                print(f"Classification report saved to: {save_path}")
            
            return report
            
        except Exception as e:
            print(f"Error generating classification report: {e}")
            return None
    
    def analyze_misclassifications(self, data_loader, predictions, labels, top_n=20):
        """Analyze top misclassified images"""
        try:
            misclassified_indices = np.where(predictions != labels)[0]
            
            if len(misclassified_indices) == 0:
                print("🎉 No misclassifications found! Perfect model!")
                return []
            
            print(f"\nAnalyzing {len(misclassified_indices)} misclassifications...")
            
            # Get image paths for misclassified items
            all_image_paths = []
            for batch_data in data_loader:
                if len(batch_data) == 3:
                    _, _, paths = batch_data
                    all_image_paths.extend(paths)
                else:
                    # If no paths, create placeholder
                    all_image_paths.extend(['N/A'] * len(batch_data[0]))
            
            # Get probabilities if available
            if hasattr(self, 'last_probabilities'):
                probs = self.last_probabilities
            else:
                probs = None
            
            # Display top N misclassifications
            display_n = min(top_n, len(misclassified_indices))
            print(f"\nTop {display_n} Misclassifications (by confidence):")
            print("="*90)
            print(f"{'Image':<25} {'True':<25} {'Predicted':<25} {'Confidence':<12}")
            print("-"*90)
            
            misclassifications = []
            for i, idx in enumerate(misclassified_indices[:display_n]):
                if idx < len(all_image_paths):
                    true_label = self.categories[labels[idx]]
                    pred_label = self.categories[predictions[idx]]
                    
                    # Get image path
                    img_path = all_image_paths[idx]
                    img_name = os.path.basename(img_path) if img_path != 'N/A' else f'Image_{idx}'
                    
                    # Get confidence if available
                    confidence = probs[idx][predictions[idx]] if probs is not None else 'N/A'
                    
                    print(f"{img_name:<25} {true_label:<25} {pred_label:<25} "
                          f"{confidence:<12.4f}" if isinstance(confidence, (float, np.floating)) else f"{confidence:<12}")
                    
                    misclassifications.append({
                        'index': idx,
                        'image_path': img_path,
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'confidence': float(confidence) if isinstance(confidence, (float, np.floating)) else None
                    })
            
            print("-"*90)
            
            # Show misclassification statistics
            print(f"\nMisclassification Statistics:")
            print(f"  Total misclassifications: {len(misclassified_indices)}/{len(labels)} "
                  f"({100.0 * len(misclassified_indices) / len(labels):.2f}%)")
            
            return misclassifications
            
        except Exception as e:
            print(f"Error analyzing misclassifications: {e}")
            return []
    
    def evaluate_model(self, save_results=False, results_dir='results'):
        """Complete model evaluation with all metrics"""
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        # Create results directory
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Run evaluation
        results = self.start_evaluation_loop()
        
        if results is None:
            print("❌ Evaluation failed!")
            return None
        
        # Store probabilities for misclassification analysis
        self.last_probabilities = results['probabilities']
        
        # Generate confusion matrix
        cm_path = None
        if save_results:
            cm_path = os.path.join(results_dir, f'confusion_matrix_{timestamp}.png')
        
        cm = self.generate_confusion_matrix(
            results['predictions'], 
            results['labels'],
            save_path=cm_path,
            show_plot=True
        )
        
        # Generate classification report
        report_path = None
        if save_results:
            report_path = os.path.join(results_dir, f'classification_report_{timestamp}.txt')
        
        report = self.generate_classification_report(
            results['predictions'], 
            results['labels'],
            save_path=report_path
        )
        
        # Analyze misclassifications
        misclassifications = self.analyze_misclassifications(
            self.data_loader,
            results['predictions'],
            results['labels'],
            top_n=15
        )
        
        # Save comprehensive results if requested
        if save_results:
            import json
            import pandas as pd
            
            # Save metrics
            metrics = {
                'accuracy': results['accuracy'],
                'average_loss': results['average_loss'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'avg_confidence': results['avg_confidence'],
                'correct': results['correct'],
                'total': results['total'],
                'timestamp': timestamp
            }
            
            metrics_path = os.path.join(results_dir, f'evaluation_metrics_{timestamp}.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save predictions
            predictions_data = []
            for i in range(len(results['labels'])):
                predictions_data.append({
                    'true_label': self.categories[results['labels'][i]],
                    'predicted_label': self.categories[results['predictions'][i]],
                    'is_correct': results['predictions'][i] == results['labels'][i],
                    'confidence': float(results['confidences'][i]) if i < len(results['confidences']) else None
                })
            
            predictions_df = pd.DataFrame(predictions_data)
            predictions_path = os.path.join(results_dir, f'predictions_{timestamp}.csv')
            predictions_df.to_csv(predictions_path, index=False)
            
            # Save misclassifications
            if misclassifications:
                misclass_df = pd.DataFrame(misclassifications)
                misclass_path = os.path.join(results_dir, f'misclassifications_{timestamp}.csv')
                misclass_df.to_csv(misclass_path, index=False)
            
            print(f"\n📊 Evaluation results saved to '{results_dir}/':")
            print(f"  - {os.path.basename(metrics_path)}")
            print(f"  - {os.path.basename(predictions_path)}")
            if misclassifications:
                print(f"  - {os.path.basename(misclass_path)}")
            if cm_path:
                print(f"  - {os.path.basename(cm_path)}")
            if report_path:
                print(f"  - {os.path.basename(report_path)}")
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
        return {
            'metrics': results,
            'confusion_matrix': cm,
            'classification_report': report,
            'misclassifications': misclassifications
        }