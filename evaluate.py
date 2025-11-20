"""
Script para evaluar un modelo entrenado.
"""

import torch
import argparse
from pathlib import Path

from src.config import Config
from src.data import get_data_loaders
from src.models import create_model
from src.evaluation import evaluate_model, plot_confusion_matrix


def main():
    """Función principal de evaluación."""
    
    # Argumentos
    parser = argparse.ArgumentParser(description='Evaluar modelo entrenado')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Ruta al modelo guardado')
    parser.add_argument('--variant', type=str, default='simple',
                       choices=['simple', 'extended'],
                       help='Variante del modelo')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                       help='Tamaño del batch')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Dispositivo')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset a evaluar')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Cargar datos
    print("\nCargando datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size
    )
    
    # Seleccionar dataset
    if args.dataset == 'train':
        data_loader = train_loader
    elif args.dataset == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    # Cargar modelo
    print(f"\nCargando modelo desde: {args.model_path}")
    model = create_model(variant=args.variant, pretrained=False)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Evaluar
    print(f"\nEvaluando en conjunto de {args.dataset}...")
    results = evaluate_model(model, data_loader, device=device)
    
    # Visualizar matriz de confusión
    print("\nGenerando matriz de confusión...")
    plot_confusion_matrix(
        results['confusion_matrix'],
        normalize=True,
        save_path=f"confusion_matrix_{args.dataset}.png"
    )
    
    print("\nEvaluación completada!")


if __name__ == "__main__":
    main()
