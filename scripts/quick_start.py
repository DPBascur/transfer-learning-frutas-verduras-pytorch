"""
Script de inicio rápido para verificar que el proyecto funciona correctamente.
Realiza una prueba básica de carga de datos y creación de modelos.
"""

import torch
from pathlib import Path

from src.config import Config
from src.data import get_data_loaders
from src.models import create_model

def main():
    """Función principal de verificación rápida."""
    
    print("=" * 80)
    print("VERIFICACIÓN RÁPIDA DEL PROYECTO")
    print("=" * 80)
    
    # 1. Verificar PyTorch y CUDA
    print("\n1. Verificando instalación de PyTorch...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Verificar estructura de directorios
    print("\n2. Verificando estructura de directorios...")
    dirs_to_check = [
        Config.DATA_DIR,
        Config.TRAIN_DIR,
        Config.VAL_DIR,
        Config.TEST_DIR,
        Config.MODELS_DIR
    ]
    
    all_exist = True
    for dir_path in dirs_to_check:
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {dir_path.name}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n   ADVERTENCIA: Algunos directorios no existen.")
        print("   Asegúrate de tener la estructura de datos correcta.")
        return
    
    # 3. Verificar clases seleccionadas
    print(f"\n3. Clases seleccionadas:")
    for i, clase in enumerate(Config.SELECTED_CLASSES, 1):
        print(f"   {i}. {clase}")
    
    # 4. Cargar un batch pequeño de datos
    print("\n4. Cargando datos de prueba...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders(batch_size=8)
        
        # Obtener un batch
        images, labels = next(iter(train_loader))
        print(f"   ✓ Batch de imágenes cargado: {images.shape}")
        print(f"   ✓ Batch de etiquetas cargado: {labels.shape}")
        print(f"   ✓ Rango de valores de imágenes: [{images.min():.2f}, {images.max():.2f}]")
        
    except Exception as e:
        print(f"   ✗ Error al cargar datos: {str(e)}")
        return
    
    # 5. Crear modelos
    print("\n5. Creando modelos...")
    
    try:
        # Modelo Versión 1
        print("\n   Versión 1 (Simple):")
        model_v1 = create_model(variant='simple', pretrained=True)
        trainable_v1, total_v1 = model_v1.get_trainable_params()
        print(f"   ✓ Modelo creado exitosamente")
        
        # Modelo Versión 2
        print("\n   Versión 2 (Extendido):")
        model_v2 = create_model(variant='extended', pretrained=True)
        trainable_v2, total_v2 = model_v2.get_trainable_params()
        print(f"   ✓ Modelo creado exitosamente")
        
    except Exception as e:
        print(f"   ✗ Error al crear modelos: {str(e)}")
        return
    
    # 6. Probar inferencia
    print("\n6. Probando inferencia...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_v1.to(device)
        model_v1.eval()
        
        with torch.no_grad():
            sample_input = images[:2].to(device)
            output = model_v1(sample_input)
            
        print(f"   ✓ Inferencia exitosa")
        print(f"   ✓ Input shape: {sample_input.shape}")
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Número de clases: {output.shape[1]}")
        
    except Exception as e:
        print(f"   ✗ Error en inferencia: {str(e)}")
        return
    
    # 7. Resumen
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print("✓ Todas las verificaciones pasaron exitosamente")
    print("\nEl proyecto está listo para comenzar el entrenamiento.")
    print("\nPróximos pasos:")
    print("  1. Ejecuta el notebook: jupyter notebook notebooks/transfer_learning_notebook.ipynb")
    print("  2. O entrena desde CLI: python train.py --variant simple --epochs 5")
    print("  3. O inicia la app: streamlit run app/app.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
