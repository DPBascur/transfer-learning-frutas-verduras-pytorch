"""
Script para comparar ambas variantes del modelo y generar reportes.
"""

import torch
import argparse
from pathlib import Path

from src.config import Config
from src.data import get_data_loaders
from src.models import create_model
from src.evaluation import evaluate_model, plot_confusion_matrix
from src.utils import ResultsSaver


def compare_models(model_path_v1: str, model_path_v2: str, 
                   experiment_name: str = "comparison"):
    """
    Compara dos variantes del modelo.
    
    Args:
        model_path_v1: Ruta al modelo Versión 1
        model_path_v2: Ruta al modelo Versión 2
        experiment_name: Nombre del experimento
    """
    print("=" * 80)
    print("COMPARACIÓN DE VARIANTES DEL MODELO")
    print("=" * 80)
    
    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDispositivo: {device}\n")
    
    # Cargar datos
    print("Cargando datos...")
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Cargar Modelo Versión 1
    print("\n" + "-" * 80)
    print("EVALUANDO VERSIÓN 1 (Simple)")
    print("-" * 80)
    
    model_v1 = create_model(variant='simple', pretrained=False)
    checkpoint_v1 = torch.load(model_path_v1, map_location=device)
    model_v1.load_state_dict(checkpoint_v1['model_state_dict'])
    model_v1.to(device)
    
    results_v1 = evaluate_model(model_v1, test_loader, device=device)
    
    # Cargar Modelo Versión 2
    print("\n" + "-" * 80)
    print("EVALUANDO VERSIÓN 2 (Extendido)")
    print("-" * 80)
    
    model_v2 = create_model(variant='extended', pretrained=False)
    checkpoint_v2 = torch.load(model_path_v2, map_location=device)
    model_v2.load_state_dict(checkpoint_v2['model_state_dict'])
    model_v2.to(device)
    
    results_v2 = evaluate_model(model_v2, test_loader, device=device)
    
    # Comparación
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 80)
    
    print(f"\n{'Métrica':<20} {'V1 (Simple)':<15} {'V2 (Extendido)':<15} {'Diferencia':<15}")
    print("-" * 70)
    
    metrics = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision_avg'),
        ('Recall', 'recall_avg'),
        ('F1-Score', 'f1_avg')
    ]
    
    for metric_name, metric_key in metrics:
        v1_val = results_v1[metric_key]
        v2_val = results_v2[metric_key]
        diff = v2_val - v1_val
        
        print(f"{metric_name:<20} {v1_val:<15.4f} {v2_val:<15.4f} {diff:+.4f}")
    
    # Análisis por clase
    print("\n" + "=" * 80)
    print("COMPARACIÓN POR CLASE (F1-Score)")
    print("=" * 80)
    
    class_names = Config.SELECTED_CLASSES
    print(f"\n{'Clase':<15} {'V1 (Simple)':<15} {'V2 (Extendido)':<15} {'Diferencia':<15}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        v1_f1 = results_v1['f1_per_class'][i]
        v2_f1 = results_v2['f1_per_class'][i]
        diff = v2_f1 - v1_f1
        
        print(f"{class_name:<15} {v1_f1:<15.4f} {v2_f1:<15.4f} {diff:+.4f}")
    
    # Guardar resultados
    print("\n" + "=" * 80)
    print("GUARDANDO RESULTADOS")
    print("=" * 80)
    
    saver = ResultsSaver()
    report_path = saver.save_comparison_report(results_v1, results_v2, experiment_name)
    
    # Guardar matrices de confusión
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # V1
    import seaborn as sns
    import numpy as np
    
    conf_v1_norm = results_v1['confusion_matrix'].astype('float') / \
                   results_v1['confusion_matrix'].sum(axis=1)[:, np.newaxis]
    sns.heatmap(conf_v1_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Versión 1 (Simple)', fontsize=14)
    axes[0].set_ylabel('Etiqueta Verdadera')
    axes[0].set_xlabel('Etiqueta Predicha')
    
    # V2
    conf_v2_norm = results_v2['confusion_matrix'].astype('float') / \
                   results_v2['confusion_matrix'].sum(axis=1)[:, np.newaxis]
    sns.heatmap(conf_v2_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Versión 2 (Extendido)', fontsize=14)
    axes[1].set_ylabel('Etiqueta Verdadera')
    axes[1].set_xlabel('Etiqueta Predicha')
    
    plt.tight_layout()
    saver.save_plot(plt.gcf(), experiment_name, 'confusion_matrices_comparison')
    plt.close()
    
    # Conclusiones
    print("\n" + "=" * 80)
    print("CONCLUSIONES")
    print("=" * 80)
    
    if results_v2['accuracy'] > results_v1['accuracy']:
        winner = "Versión 2 (Extendido)"
        improvement = (results_v2['accuracy'] - results_v1['accuracy']) * 100
        print(f"\n✓ La {winner} muestra mejor desempeño")
        print(f"  Mejora en accuracy: +{improvement:.2f}%")
        print(f"\n  Esto sugiere que:")
        print(f"  - Batch Normalization ayuda a estabilizar el entrenamiento")
        print(f"  - Dropout reduce el overfitting")
        print(f"  - Las capas adicionales permiten mejor representación")
    elif results_v1['accuracy'] > results_v2['accuracy']:
        winner = "Versión 1 (Simple)"
        degradation = (results_v1['accuracy'] - results_v2['accuracy']) * 100
        print(f"\n✓ La {winner} muestra mejor desempeño")
        print(f"  Ventaja en accuracy: +{degradation:.2f}%")
        print(f"\n  Esto sugiere que:")
        print(f"  - El modelo simple es suficiente para este problema")
        print(f"  - La arquitectura extendida podría estar sobreajustando")
        print(f"  - Se requiere más data augmentation en V2")
    else:
        print(f"\n≈ Ambas versiones muestran desempeño similar")
        print(f"  Diferencia en accuracy: {abs(results_v2['accuracy'] - results_v1['accuracy'])*100:.2f}%")
    
    print("\n" + "=" * 80)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Comparar variantes del modelo')
    parser.add_argument('--model-v1', type=str, required=True,
                       help='Ruta al modelo Versión 1')
    parser.add_argument('--model-v2', type=str, required=True,
                       help='Ruta al modelo Versión 2')
    parser.add_argument('--name', type=str, default='comparison',
                       help='Nombre del experimento')
    
    args = parser.parse_args()
    
    compare_models(args.model_v1, args.model_v2, args.name)


if __name__ == "__main__":
    main()
