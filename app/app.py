"""
Aplicación Streamlit para predicción de frutas y verduras.
Interfaz gráfica para interactuar con el modelo entrenado.
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Añadir el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.config import Config
from src.models import create_model
from src.data import get_transforms


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


def main():
    """
    Función principal de la aplicación Streamlit.
    """
    st.set_page_config(
        page_title="Clasificador de Frutas y Verduras",
        page_icon="🍎",
        layout="wide"
    )
    
    st.title("Clasificador de Frutas y Verduras")
    st.markdown("### Transfer Learning con MobileNetV3")
    
    # Sidebar para configuración
    st.sidebar.header("Configuración")
    
    # Seleccionar variante del modelo
    variant = st.sidebar.selectbox(
        "Seleccionar Variante del Modelo",
        ["simple", "extended"],
        help="Simple: clasificador básico | Extended: clasificador con capas ocultas"
    )
    
    # Selector de modelo
    models_dir = Config.MODELS_DIR
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        if model_files:
            selected_model = st.sidebar.selectbox(
                "Seleccionar Modelo",
                model_files,
                format_func=lambda x: x.name
            )
        else:
            st.sidebar.warning("No se encontraron modelos guardados.")
            selected_model = None
    else:
        st.sidebar.warning(f"Directorio de modelos no existe: {models_dir}")
        selected_model = None
    
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
        """)
    
    # Contenido principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Cargar Imagen")
        
        # Opciones de carga
        upload_option = st.radio(
            "Método de carga:",
            ["Subir archivo", "Usar cámara"]
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
        st.header("Predicción")
        
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
        
        elif selected_model is None:
            st.info("Por favor, entrena un modelo primero o carga un modelo guardado.")
        else:
            st.info("Sube una imagen para comenzar la clasificación.")
    
    # Información adicional
    st.markdown("---")
    st.markdown("""
    ### Instrucciones de Uso
    1. Selecciona la variante del modelo en el panel lateral
    2. Elige un modelo entrenado del selector
    3. Carga una imagen (subir archivo o usar cámara)
    4. Haz clic en "Clasificar Imagen"
    5. Observa la predicción y las probabilidades
    """)


if __name__ == "__main__":
    main()
