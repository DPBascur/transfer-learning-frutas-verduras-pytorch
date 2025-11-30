"""
Utilidades para guardar resultados, métricas y visualizaciones.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import torch
import matplotlib.pyplot as plt


class ResultsSaver:
    """
    Clase para guardar resultados de experimentos de forma organizada.
    """
    
    def __init__(self, base_dir: Path = None):
        """
        Args:
            base_dir: Directorio base para guardar resultados
        """
        self.base_dir = base_dir or Path('results')
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear subdirectorios
        self.metrics_dir = self.base_dir / 'metrics'
        self.plots_dir = self.base_dir / 'plots'
        self.models_dir = self.base_dir / 'models'
        self.reports_dir = self.base_dir / 'reports'
        
        for dir_path in [self.metrics_dir, self.plots_dir, self.models_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_metrics(self, metrics: Dict[str, Any], experiment_name: str):
        """
        Guarda métricas en formato JSON.
        
        Args:
            metrics: Diccionario con métricas
            experiment_name: Nombre del experimento
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}_metrics.json"
        filepath = self.metrics_dir / filename
        
        # Convertir numpy arrays a listas para JSON
        metrics_serializable = self._make_serializable(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=4)
        
        print(f"Métricas guardadas en: {filepath}")
        return filepath
    
    def save_history(self, history: Dict[str, list], experiment_name: str):
        """
        Guarda historial de entrenamiento.
        
        Args:
            history: Diccionario con historial (loss, accuracy, etc.)
            experiment_name: Nombre del experimento
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}_history.pkl"
        filepath = self.metrics_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(history, f)
        
        print(f"Historial guardado en: {filepath}")
        return filepath
    
    def save_plot(self, fig: plt.Figure, experiment_name: str, plot_type: str):
        """
        Guarda una figura de matplotlib.
        
        Args:
            fig: Figura de matplotlib
            experiment_name: Nombre del experimento
            plot_type: Tipo de gráfico (e.g., 'confusion_matrix', 'loss_curves')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{plot_type}_{timestamp}.png"
        filepath = self.plots_dir / filename
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {filepath}")
        return filepath
    
    def save_comparison_report(self, results_v1: Dict, results_v2: Dict, 
                               experiment_name: str):
        """
        Guarda un reporte comparativo entre dos variantes.
        
        Args:
            results_v1: Resultados de la versión 1
            results_v2: Resultados de la versión 2
            experiment_name: Nombre del experimento
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_comparison_{timestamp}.txt"
        filepath = self.reports_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE COMPARATIVO DE VARIANTES\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experimento: {experiment_name}\n\n")
            
            f.write("VERSIÓN 1 (Simple)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {results_v1['accuracy']:.4f}\n")
            f.write(f"Precision: {results_v1['precision_avg']:.4f}\n")
            f.write(f"Recall: {results_v1['recall_avg']:.4f}\n")
            f.write(f"F1-Score: {results_v1['f1_avg']:.4f}\n\n")
            
            f.write("VERSIÓN 2 (Extendido)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {results_v2['accuracy']:.4f}\n")
            f.write(f"Precision: {results_v2['precision_avg']:.4f}\n")
            f.write(f"Recall: {results_v2['recall_avg']:.4f}\n")
            f.write(f"F1-Score: {results_v2['f1_avg']:.4f}\n\n")
            
            f.write("DIFERENCIAS\n")
            f.write("-" * 80 + "\n")
            diff_acc = results_v2['accuracy'] - results_v1['accuracy']
            diff_prec = results_v2['precision_avg'] - results_v1['precision_avg']
            diff_rec = results_v2['recall_avg'] - results_v1['recall_avg']
            diff_f1 = results_v2['f1_avg'] - results_v1['f1_avg']
            
            f.write(f"Δ Accuracy: {diff_acc:+.4f} ({diff_acc*100:+.2f}%)\n")
            f.write(f"Δ Precision: {diff_prec:+.4f}\n")
            f.write(f"Δ Recall: {diff_rec:+.4f}\n")
            f.write(f"Δ F1-Score: {diff_f1:+.4f}\n\n")
            
            f.write("CONCLUSIÓN\n")
            f.write("-" * 80 + "\n")
            if diff_acc > 0:
                f.write("La Versión 2 (Extendido) muestra mejor desempeño.\n")
            elif diff_acc < 0:
                f.write("La Versión 1 (Simple) muestra mejor desempeño.\n")
            else:
                f.write("Ambas versiones muestran desempeño similar.\n")
        
        print(f"Reporte comparativo guardado en: {filepath}")
        return filepath
    
    def _make_serializable(self, obj):
        """Convierte objetos numpy a tipos serializables."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._make_serializable(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def save_experiment_summary(results: Dict, history: Dict, 
                            experiment_name: str, variant: str):
    """
    Función auxiliar para guardar un resumen completo del experimento.
    
    Args:
        results: Resultados de evaluación
        history: Historial de entrenamiento
        experiment_name: Nombre del experimento
        variant: Variante del modelo ('simple' o 'extended')
    """
    saver = ResultsSaver()
    
    # Guardar métricas
    saver.save_metrics(results, f"{experiment_name}_{variant}")
    
    # Guardar historial
    saver.save_history(history, f"{experiment_name}_{variant}")
    
    # Crear resumen en texto
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = saver.reports_dir / f"{experiment_name}_{variant}_{timestamp}_summary.txt"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RESUMEN DEL EXPERIMENTO: {experiment_name}\n")
        f.write(f"Variante: {variant}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RESULTADOS FINALES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision_avg']:.4f}\n")
        f.write(f"Recall: {results['recall_avg']:.4f}\n")
        f.write(f"F1-Score: {results['f1_avg']:.4f}\n\n")
        
        f.write("ENTRENAMIENTO\n")
        f.write("-" * 80 + "\n")
        f.write(f"Epochs completados: {len(history['train_loss'])}\n")
        f.write(f"Mejor val_loss: {min(history['val_loss']):.4f}\n")
        f.write(f"Mejor val_acc: {max(history['val_acc']):.2f}%\n")
    
    print(f"Resumen guardado en: {filepath}")
    return filepath
