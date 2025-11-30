"""
Aplicación Streamlit completa para clasificación de frutas y verduras.
Incluye entrenamiento, evaluación y predicción de modelos.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time

# Añadir el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.config import Config
from src.models import create_model
from src.data import get_data_loaders, get_transforms
from src.training import Trainer, EarlyStopping
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_training_history
from src.utils import ResultsSaver, create_performance_comparison_plot


class FruitVegetableClassifier:
    """
    Clase para manejar la predicción con el modelo.
    """
    
    def __init__(self, model_path: str, variant: str = 'simple', device: str = 'cpu'):
        """
        Args:
            model_path: Ruta al modelo guardado
            variant: Variante del modelo ('simple' o 'extended')
            device: Dispositivo ('cuda' o 'cpu')
        """
        self.device = device
        self.variant = variant
        self.class_names = Config.SELECTED_CLASSES
        
        # Cargar modelo
        self.model = self._load_model(model_path, variant)
        self.transform = get_transforms(augment=False)
    
    def _load_model(self, model_path: str, variant: str):
        """Carga el modelo desde un checkpoint."""
        model = create_model(variant=variant, pretrained=False)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image: Image.Image):
        """
        Realiza una predicción sobre una imagen.
        
        Args:
            image: Imagen PIL
            
        Returns:
            Tupla (clase_predicha, probabilidades)
        """
        # Preprocesar imagen
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predicción
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = probabilities.argmax(1).item()
        
        predicted_class = self.class_names[predicted_idx]
        probs = probabilities[0].cpu().numpy()
        
        return predicted_class, probs


def train_model_ui():
    """Interfaz para entrenar modelos."""
    st.header("Entrenamiento de Modelos")
    
    # Inicializar session_state para controlar el estado de entrenamiento
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    if 'cancel_training' not in st.session_state:
        st.session_state.cancel_training = False
    if 'training_params' not in st.session_state:
        st.session_state.training_params = None
    
    # Verificar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.info(f"Dispositivo disponible: **{device.upper()}**")
    
    # Si el entrenamiento está en progreso, mostrar solo controles de monitoreo
    if st.session_state.training_in_progress:
        st.warning("⚠️ Entrenamiento en progreso. Los controles están deshabilitados.")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("❌ Cancelar Entrenamiento", type="secondary", use_container_width=True):
                st.session_state.cancel_training = True
                st.session_state.training_in_progress = False
                st.success("Entrenamiento cancelado por el usuario")
                st.rerun()
        
        # Recuperar parámetros guardados
        if st.session_state.training_params:
            params = st.session_state.training_params
            variant = params['variant']
            model_name = params['model_name']
            use_pretrained = params['use_pretrained']
            num_epochs = params['num_epochs']
            batch_size = params['batch_size']
            learning_rate = params['learning_rate']
            patience = params['patience']
        else:
            st.error("Error: No se encontraron parámetros de entrenamiento")
            st.session_state.training_in_progress = False
            st.rerun()
    else:
        # Configuración de entrenamiento (solo visible cuando NO está entrenando)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configuración del Modelo")
            variant = st.selectbox(
                "Variante del Modelo",
                ["simple", "extended"],
                help="Simple: clasificador básico | Extended: clasificador con capas ocultas",
                key="variant_select"
            )
            
            model_name = st.text_input(
                "Nombre del Modelo",
                value=f"mobilenet_v3_{variant}",
                help="Nombre para guardar el modelo",
                key="model_name_input"
            )
            
            use_pretrained = st.checkbox(
                "Usar pesos preentrenados (ImageNet)", 
                value=True,
                key="pretrained_check"
            )
        
        with col2:
            st.subheader("Hiperparámetros")
            num_epochs = st.slider(
                "Número de Epochs", 
                min_value=1, 
                max_value=100, 
                value=20,
                key="epochs_slider"
            )
            batch_size = st.slider(
                "Batch Size", 
                min_value=8, 
                max_value=64, 
                value=32, 
                step=8,
                key="batch_slider"
            )
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001,
                key="lr_slider"
            )
            patience = st.slider(
                "Patience (Early Stopping)", 
                min_value=3, 
                max_value=20, 
                value=10,
                key="patience_slider"
            )
    
    # Botón de inicio de entrenamiento (solo visible cuando NO está entrenando)
    if not st.session_state.training_in_progress:
        if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
            # Guardar parámetros en session_state
            st.session_state.training_params = {
                'variant': variant,
                'model_name': model_name,
                'use_pretrained': use_pretrained,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'patience': patience
            }
            # Marcar entrenamiento como en progreso
            st.session_state.training_in_progress = True
            st.session_state.cancel_training = False
            st.rerun()
    
    # Ejecutar entrenamiento si está marcado como en progreso
    if st.session_state.training_in_progress and st.session_state.training_params:
        
        # Recuperar parámetros
        params = st.session_state.training_params
        variant = params['variant']
        model_name = params['model_name']
        use_pretrained = params['use_pretrained']
        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        patience = params['patience']
        
        # Crear placeholder para progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
        
        try:
            status_text.text("Cargando datos...")
            train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)
            
            status_text.text("Creando modelo...")
            model = create_model(variant=variant, pretrained=use_pretrained)
            
            # Configurar entrenamiento
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            early_stopping = EarlyStopping(patience=patience, min_delta=0.001, mode='min')
            
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
            
            status_text.text("Entrenando modelo...")
            
            # Entrenamiento con actualización de progreso
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []}
            
            for epoch in range(1, num_epochs + 1):
                # Verificar cancelación
                if st.session_state.cancel_training:
                    status_text.text("❌ Entrenamiento cancelado por el usuario")
                    st.warning(f"Entrenamiento detenido en epoch {epoch}/{num_epochs}")
                    break
                
                # Entrenar epoch
                train_loss, train_acc = trainer.train_epoch()
                val_loss, val_acc = trainer.validate()
                
                # Actualizar historial
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
                # Actualizar interfaz
                progress = epoch / num_epochs
                progress_bar.progress(progress)
                
                status_text.text(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Mostrar métricas actuales
                metrics_df = pd.DataFrame({
                    'Métrica': ['Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'],
                    'Valor': [f"{train_loss:.4f}", f"{train_acc:.2f}%", f"{val_loss:.4f}", f"{val_acc:.2f}%"]
                })
                metrics_placeholder.dataframe(metrics_df, hide_index=True)
                
                # Scheduler
                if scheduler:
                    scheduler.step()
                
                # Guardar mejor modelo
                if val_acc > trainer.best_val_acc:
                    trainer.best_val_acc = val_acc
                    trainer.save_checkpoint(epoch, model_name, is_best=True)
                
                # Early stopping
                if early_stopping(val_loss, epoch):
                    status_text.text(f"Early stopping en epoch {epoch}")
                    break
            
            trainer.history = history
            
            # Solo mostrar resultados si no fue cancelado
            if not st.session_state.cancel_training:
                st.success("✅ Entrenamiento completado!")
                
                # Mostrar resultados finales
                st.subheader("Resultados del Entrenamiento")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Train Loss Final", f"{history['train_loss'][-1]:.4f}")
                col2.metric("Train Acc Final", f"{history['train_acc'][-1]:.2f}%")
                col3.metric("Val Loss Final", f"{history['val_loss'][-1]:.4f}")
                col4.metric("Val Acc Final", f"{history['val_acc'][-1]:.2f}%")
                
                # Gráfico de curvas
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                epochs_range = range(1, len(history['train_loss']) + 1)
                
                axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
                axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Curva de Pérdida')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(epochs_range, history['train_acc'], 'b-', label='Train Acc')
                axes[1].plot(epochs_range, history['val_acc'], 'r-', label='Val Acc')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy (%)')
                axes[1].set_title('Curva de Accuracy')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Evaluar en test
                st.subheader("Evaluación en Conjunto de Prueba")
                with st.spinner("Evaluando modelo..."):
                    results = evaluate_model(model, test_loader, device=device)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                    col2.metric("Precision", f"{results['precision_avg']:.4f}")
                    col3.metric("Recall", f"{results['recall_avg']:.4f}")
                    col4.metric("F1-Score", f"{results['f1_avg']:.4f}")
                    
                    # Mostrar matriz de confusión
                    fig, ax = plt.subplots(figsize=(8, 6))
                    import seaborn as sns
                    conf_matrix_norm = results['confusion_matrix'].astype('float') / \
                                      results['confusion_matrix'].sum(axis=1)[:, np.newaxis]
                    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                               xticklabels=Config.SELECTED_CLASSES,
                               yticklabels=Config.SELECTED_CLASSES, ax=ax)
                    ax.set_title('Matriz de Confusión Normalizada')
                    ax.set_ylabel('Etiqueta Verdadera')
                    ax.set_xlabel('Etiqueta Predicha')
                    st.pyplot(fig)
                    plt.close()
            
            # Restablecer estado de entrenamiento
            st.session_state.training_in_progress = False
            st.session_state.cancel_training = False
            st.session_state.training_params = None
            
        except Exception as e:
            st.error(f"Error durante el entrenamiento: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            # Restablecer estado en caso de error
            st.session_state.training_in_progress = False
            st.session_state.cancel_training = False
            st.session_state.training_params = None


def evaluate_model_ui():
    """Interfaz para evaluar modelos."""
    st.header("Evaluación de Modelos")
    
    models_dir = Config.MODELS_DIR
    
    if not models_dir.exists() or not list(models_dir.glob("*.pth")):
        st.warning("No se encontraron modelos entrenados. Por favor, entrena un modelo primero.")
        return
    
    model_files = list(models_dir.glob("*.pth"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Seleccionar Modelo",
            model_files,
            format_func=lambda x: x.name
        )
        
        variant = st.selectbox(
            "Variante del Modelo",
            ["simple", "extended"]
        )
    
    with col2:
        dataset_choice = st.selectbox(
            "Conjunto de Datos",
            ["test", "validation", "train"]
        )
        
        batch_size = st.slider("Batch Size", 8, 64, 32, 8)
    
    if st.button("Evaluar Modelo", type="primary"):
        with st.spinner("Evaluando modelo..."):
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Cargar datos
                train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)
                
                data_loader = {
                    'train': train_loader,
                    'validation': val_loader,
                    'test': test_loader
                }[dataset_choice]
                
                # Cargar modelo
                model = create_model(variant=variant, pretrained=False)
                checkpoint = torch.load(selected_model, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                
                # Evaluar
                results = evaluate_model(model, data_loader, device=device)
                
                # Mostrar métricas globales
                st.subheader("Métricas Globales")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                col2.metric("Precision", f"{results['precision_avg']:.4f}")
                col3.metric("Recall", f"{results['recall_avg']:.4f}")
                col4.metric("F1-Score", f"{results['f1_avg']:.4f}")
                
                # Métricas por clase
                st.subheader("Métricas por Clase")
                metrics_df = pd.DataFrame({
                    'Clase': Config.SELECTED_CLASSES,
                    'Precision': results['precision_per_class'],
                    'Recall': results['recall_per_class'],
                    'F1-Score': results['f1_per_class'],
                    'Support': results['support_per_class']
                })
                st.dataframe(metrics_df, hide_index=True)
                
                # Matriz de confusión
                st.subheader("Matriz de Confusión")
                fig, ax = plt.subplots(figsize=(10, 8))
                import seaborn as sns
                conf_matrix_norm = results['confusion_matrix'].astype('float') / \
                                  results['confusion_matrix'].sum(axis=1)[:, np.newaxis]
                sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=Config.SELECTED_CLASSES,
                           yticklabels=Config.SELECTED_CLASSES, ax=ax)
                ax.set_title('Matriz de Confusión Normalizada')
                ax.set_ylabel('Etiqueta Verdadera')
                ax.set_xlabel('Etiqueta Predicha')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error durante la evaluación: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def compare_models_ui():
    """Interfaz para comparar dos modelos."""
    st.header("Comparación de Modelos")
    
    models_dir = Config.MODELS_DIR
    
    if not models_dir.exists() or len(list(models_dir.glob("*.pth"))) < 2:
        st.warning("Se necesitan al menos 2 modelos entrenados para comparar.")
        return
    
    model_files = list(models_dir.glob("*.pth"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Modelo 1")
        model1 = st.selectbox(
            "Seleccionar Modelo 1",
            model_files,
            format_func=lambda x: x.name,
            key="model1"
        )
        variant1 = st.selectbox("Variante", ["simple", "extended"], key="variant1")
    
    with col2:
        st.subheader("Modelo 2")
        model2 = st.selectbox(
            "Seleccionar Modelo 2",
            model_files,
            format_func=lambda x: x.name,
            key="model2"
        )
        variant2 = st.selectbox("Variante", ["simple", "extended"], key="variant2")
    
    if st.button("Comparar Modelos", type="primary"):
        with st.spinner("Comparando modelos..."):
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Cargar datos
                _, _, test_loader = get_data_loaders()
                
                # Evaluar Modelo 1
                st.info("Evaluando Modelo 1...")
                model_1 = create_model(variant=variant1, pretrained=False)
                checkpoint_1 = torch.load(model1, map_location=device)
                model_1.load_state_dict(checkpoint_1['model_state_dict'])
                model_1.to(device)
                results_1 = evaluate_model(model_1, test_loader, device=device)
                
                # Evaluar Modelo 2
                st.info("Evaluando Modelo 2...")
                model_2 = create_model(variant=variant2, pretrained=False)
                checkpoint_2 = torch.load(model2, map_location=device)
                model_2.load_state_dict(checkpoint_2['model_state_dict'])
                model_2.to(device)
                results_2 = evaluate_model(model_2, test_loader, device=device)
                
                # Comparación
                st.subheader("Comparación de Métricas")
                
                comparison_df = pd.DataFrame({
                    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Modelo 1': [
                        f"{results_1['accuracy']:.4f}",
                        f"{results_1['precision_avg']:.4f}",
                        f"{results_1['recall_avg']:.4f}",
                        f"{results_1['f1_avg']:.4f}"
                    ],
                    'Modelo 2': [
                        f"{results_2['accuracy']:.4f}",
                        f"{results_2['precision_avg']:.4f}",
                        f"{results_2['recall_avg']:.4f}",
                        f"{results_2['f1_avg']:.4f}"
                    ],
                    'Diferencia': [
                        f"{results_2['accuracy'] - results_1['accuracy']:+.4f}",
                        f"{results_2['precision_avg'] - results_1['precision_avg']:+.4f}",
                        f"{results_2['recall_avg'] - results_1['recall_avg']:+.4f}",
                        f"{results_2['f1_avg'] - results_1['f1_avg']:+.4f}"
                    ]
                })
                
                st.dataframe(comparison_df, hide_index=True)
                
                # Gráfico comparativo
                create_performance_comparison_plot(results_1, results_2)
                st.pyplot(plt.gcf())
                plt.close()
                
                # Matrices de confusión lado a lado
                st.subheader("Matrices de Confusión")
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                import seaborn as sns
                
                conf_1_norm = results_1['confusion_matrix'].astype('float') / \
                             results_1['confusion_matrix'].sum(axis=1)[:, np.newaxis]
                sns.heatmap(conf_1_norm, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=Config.SELECTED_CLASSES,
                           yticklabels=Config.SELECTED_CLASSES, ax=axes[0])
                axes[0].set_title(f'Modelo 1: {model1.stem}')
                axes[0].set_ylabel('Etiqueta Verdadera')
                axes[0].set_xlabel('Etiqueta Predicha')
                
                conf_2_norm = results_2['confusion_matrix'].astype('float') / \
                             results_2['confusion_matrix'].sum(axis=1)[:, np.newaxis]
                sns.heatmap(conf_2_norm, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=Config.SELECTED_CLASSES,
                           yticklabels=Config.SELECTED_CLASSES, ax=axes[1])
                axes[1].set_title(f'Modelo 2: {model2.stem}')
                axes[1].set_ylabel('Etiqueta Verdadera')
                axes[1].set_xlabel('Etiqueta Predicha')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error durante la comparación: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def predict_image_ui():
    """Interfaz para predicción de imágenes."""
    st.header("Predicción de Imágenes")
    
    models_dir = Config.MODELS_DIR
    
    if not models_dir.exists() or not list(models_dir.glob("*.pth")):
        st.warning("No se encontraron modelos entrenados. Por favor, entrena un modelo primero.")
        return
    
    model_files = list(models_dir.glob("*.pth"))
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuración")
        
        variant = st.selectbox(
            "Variante del Modelo",
            ["simple", "extended"],
            help="Simple: clasificador básico | Extended: clasificador con capas ocultas"
        )
        
        selected_model = st.selectbox(
            "Seleccionar Modelo",
            model_files,
            format_func=lambda x: x.name
        )
        
        st.subheader("Cargar Imagen")
        
        upload_option = st.radio(
            "Método de carga:",
            ["Subir archivo", "Usar cámara"],
            key="predict_upload_method"
        )
        
        uploaded_file = None
        
        if upload_option == "Subir archivo":
            uploaded_file = st.file_uploader(
                "Selecciona una imagen",
                type=["jpg", "jpeg", "png"],
                help="Sube una imagen de una fruta o verdura"
            )
        else:
            uploaded_file = st.camera_input("Tomar foto")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Imagen cargada", use_column_width=True)
    
    with col2:
        st.subheader("Resultado de Predicción")
        
        if uploaded_file is not None and selected_model is not None:
            if st.button("Clasificar Imagen", type="primary"):
                with st.spinner("Procesando imagen..."):
                    try:
                        # Cargar clasificador
                        classifier = FruitVegetableClassifier(
                            model_path=str(selected_model),
                            variant=variant,
                            device='cpu'
                        )
                        
                        # Realizar predicción
                        predicted_class, probabilities = classifier.predict(image)
                        
                        # Mostrar resultados
                        st.success(f"Predicción: **{predicted_class.upper()}**")
                        st.metric(
                            label="Confianza",
                            value=f"{probabilities[Config.SELECTED_CLASSES.index(predicted_class)]:.2%}"
                        )
                        
                        # Mostrar todas las probabilidades
                        st.subheader("Probabilidades por Clase")
                        prob_dict = {
                            clase: float(prob)
                            for clase, prob in zip(Config.SELECTED_CLASSES, probabilities)
                        }
                        
                        # Ordenar por probabilidad
                        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        for clase, prob in sorted_probs:
                            st.progress(prob, text=f"{clase}: {prob:.2%}")
                        
                    except Exception as e:
                        st.error(f"Error al realizar la predicción: {str(e)}")
        else:
            st.info("Sube una imagen para comenzar la clasificación.")


def main():
    """Función principal de la aplicación."""
    st.set_page_config(
        page_title="Transfer Learning - Frutas y Verduras",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar session_state
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    
    st.title("Transfer Learning para Clasificación de Frutas y Verduras")
    st.markdown("### MobileNetV3 - Proyecto INFO1185")
    
    # Sidebar con navegación
    st.sidebar.title("Navegación")
    
    # Mostrar advertencia si hay entrenamiento en progreso
    if st.session_state.training_in_progress:
        st.sidebar.warning("⚠️ Entrenamiento en progreso. Los controles están deshabilitados.")
    
    page = st.sidebar.radio(
        "Selecciona una opción:",
        ["Predicción", "Entrenamiento", "Evaluación", "Comparación", "Información"],
        key="main_navigation",
        disabled=st.session_state.training_in_progress
    )
    
    st.sidebar.markdown("---")
    
    # Información del sistema
    st.sidebar.subheader("Estado del Sistema")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"**Dispositivo:** {device.upper()}")
    
    if torch.cuda.is_available():
        st.sidebar.success(f"**GPU:** {torch.cuda.get_device_name(0)}")
    
    models_dir = Config.MODELS_DIR
    if models_dir.exists():
        num_models = len(list(models_dir.glob("*.pth")))
        st.sidebar.metric("Modelos Entrenados", num_models)
    
    st.sidebar.markdown("---")
    
    # Información del proyecto
    with st.sidebar.expander("Información del Proyecto"):
        st.markdown("""
        **Clases disponibles:**
        - Apple (Manzana)
        - Pomegranate (Granada)
        - Mango
        - Lemon (Limón)
        - Orange (Naranja)
        
        **Modelo:** MobileNetV3 Large
        
        **Técnicas:**
        - Transfer Learning
        - Data Augmentation
        - Early Stopping
        - Batch Normalization
        - Dropout
        """)
    
    # Mostrar página seleccionada
    st.markdown("---")
    
    if page == "Predicción":
        predict_image_ui()
    elif page == "Entrenamiento":
        train_model_ui()
    elif page == "Evaluación":
        evaluate_model_ui()
    elif page == "Comparación":
        compare_models_ui()
    elif page == "Información":
        st.header("Información del Proyecto")
        st.markdown("""
        ## Transfer Learning para Clasificación de Frutas y Verduras
        
        Este proyecto implementa dos variantes de clasificadores basados en Transfer Learning con MobileNetV3:
        
        ### Versión 1 (Simple)
        - Una capa Fully Connected
        - Sin Batch Normalization
        - Sin Dropout
        
        ### Versión 2 (Extendido)
        - Arquitectura tipo embudo: 512 → 256 → 128 → 5
        - Batch Normalization después de cada capa lineal
        - Dropout con probabilidades incrementales (0.2 a 0.5)
        - Activación: ReLU
        
        ### Funcionalidades de la Aplicación
        
        1. **Predicción**: Clasifica imágenes usando modelos entrenados
        2. **Entrenamiento**: Entrena nuevos modelos con configuración personalizada
        3. **Evaluación**: Evalúa el desempeño de modelos en diferentes conjuntos de datos
        4. **Comparación**: Compara dos modelos lado a lado
        
        ### Uso Recomendado
        
        1. Entrena ambas variantes del modelo (simple y extended)
        2. Evalúa su desempeño en el conjunto de prueba
        3. Compara los resultados para analizar el impacto de BN y Dropout
        4. Usa la predicción para probar con nuevas imágenes
        
        ### Autores
        
        Proyecto desarrollado para el curso INFO1185 - Inteligencia Artificial
        """)


if __name__ == "__main__":
    main()
