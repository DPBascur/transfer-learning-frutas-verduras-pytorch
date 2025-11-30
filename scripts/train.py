"""
Script principal para entrenar el modelo.
Ejecuta el pipeline completo de entrenamiento.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse

from src.config import Config
from src.data import get_data_loaders
from src.models import create_model
from src.training import Trainer, EarlyStopping
from src.evaluation import evaluate_model, plot_training_history


def main():
    """Función principal de entrenamiento."""
    
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar modelo de clasificación')
    parser.add_argument('--variant', type=str, default='simple', 
                       choices=['simple', 'extended'],
                       help='Variante del modelo')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                       help='Número de epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                       help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Dispositivo de entrenamiento')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Deshabilitar early stopping')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Crear directorios
    Config.create_dirs()
    
    # Cargar datos
    print("\nCargando datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size
    )
    
    # Crear modelo
    print("\nCreando modelo...")
    model = create_model(variant=args.variant, pretrained=True)
    
    # Configurar entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Early Stopping
    early_stopping = None
    if not args.no_early_stopping:
        early_stopping = EarlyStopping(
            patience=Config.PATIENCE,
            min_delta=Config.MIN_DELTA,
            mode='min'
        )
    
    # Entrenar
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping=early_stopping,
        save_dir=Config.MODELS_DIR
    )
    
    model_name = f"mobilenet_v3_{args.variant}"
    history = trainer.train(num_epochs=args.epochs, model_name=model_name)
    
    # Visualizar curvas de aprendizaje
    print("\nGenerando visualizaciones...")
    plot_training_history(
        history,
        save_path=Config.MODELS_DIR / f"{model_name}_history.png"
    )
    
    # Evaluar en conjunto de prueba
    print("\nEvaluando modelo en conjunto de prueba...")
    results = evaluate_model(model, test_loader, device=device)
    
    # Guardar resultados
    print(f"\nModelo guardado en: {trainer.best_model_path}")
    print("Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()
