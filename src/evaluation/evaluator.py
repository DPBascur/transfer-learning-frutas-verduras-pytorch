"""
Módulo para evaluación de modelos con métricas y matriz de confusión.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report

from src.config import Config


class Evaluator:
    """
    Clase para evaluar modelos de clasificación.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Modelo a evaluar
            device: Dispositivo ('cuda' o 'cpu')
        """
        self.model = model.to(device)
        self.device = device
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """
        Evalúa el modelo en un dataset.
        
        Args:
            data_loader: DataLoader con los datos
            
        Returns:
            Diccionario con métricas de evaluación
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calcular métricas
        results = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        return results
    
    def _calculate_metrics(self, 
                          labels: List[int], 
                          predictions: List[int],
                          probabilities: List[np.ndarray]) -> Dict:
        """
        Calcula métricas de evaluación.
        
        Args:
            labels: Etiquetas verdaderas
            predictions: Predicciones del modelo
            probabilities: Probabilidades por clase
            
        Returns:
            Diccionario con métricas
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Accuracy global
        accuracy = accuracy_score(labels, predictions)
        
        # Métricas por clase
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Métricas promedio
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(labels, predictions)
        
        # Reporte de clasificación
        class_names = Config.SELECTED_CLASSES
        report = classification_report(
            labels, predictions, 
            target_names=class_names,
            zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'precision_avg': precision_avg,
            'recall_avg': recall_avg,
            'f1_avg': f1_avg,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }
        
        return results
    
    def print_results(self, results: Dict):
        """
        Imprime los resultados de evaluación.
        
        Args:
            results: Diccionario con resultados
        """
        print("\n" + "=" * 80)
        print("RESULTADOS DE EVALUACIÓN")
        print("=" * 80)
        
        print(f"\nAccuracy Global: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        print(f"\nMétricas Promedio (Weighted):")
        print(f"  Precision: {results['precision_avg']:.4f}")
        print(f"  Recall: {results['recall_avg']:.4f}")
        print(f"  F1-Score: {results['f1_avg']:.4f}")
        
        print(f"\nMétricas por Clase:")
        class_names = Config.SELECTED_CLASSES
        print(f"{'Clase':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 60)
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<15} "
                  f"{results['precision_per_class'][i]:>10.4f} "
                  f"{results['recall_per_class'][i]:>10.4f} "
                  f"{results['f1_per_class'][i]:>10.4f} "
                  f"{results['support_per_class'][i]:>10}")
        
        print("\n" + "=" * 80)
        print("Reporte de Clasificación Completo:")
        print("=" * 80)
        print(results['classification_report'])
        
        print(f"\nMétricas por Clase:")
        class_names = Config.SELECTED_CLASSES
        print(f"{'Clase':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 60)
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<15} "
                  f"{results['precision_per_class'][i]:>10.4f} "
                  f"{results['recall_per_class'][i]:>10.4f} "
                  f"{results['f1_per_class'][i]:>10.4f} "
                  f"{results['support_per_class'][i]:>10}")
        
        print("\n" + "=" * 80)
        print("Reporte de Clasificación Completo:")
        print("=" * 80)
        print(results['classification_report'])


def evaluate_model(model: nn.Module, 
                   data_loader: DataLoader,
                   device: str = 'cuda') -> Dict:
    """
    Función auxiliar para evaluar un modelo.
    
    Args:
        model: Modelo a evaluar
        data_loader: DataLoader con los datos
        device: Dispositivo
        
    Returns:
        Diccionario con resultados
    """
    evaluator = Evaluator(model, device)
    results = evaluator.evaluate(data_loader)
    evaluator.print_results(results)
    
    return results
