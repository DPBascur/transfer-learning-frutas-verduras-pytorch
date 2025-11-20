"""
Módulo de modelos basado en MobileNetV3 con Transfer Learning.
Implementa dos variantes: clasificador simple y clasificador extendido.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List, Tuple

from src.config import Config


class MobileNetClassifier(nn.Module):
    """
    Clasificador basado en MobileNetV3 con Transfer Learning.
    Soporta dos variantes: simple y extendido tipo embudo.
    """
    
    def __init__(self, 
                 num_classes: int,
                 variant: str = 'simple',
                 pretrained: bool = True):
        """
        Args:
            num_classes: Número de clases de salida
            variant: 'simple' o 'extended'
            pretrained: Si True, usa pesos preentrenados de ImageNet
        """
        super(MobileNetClassifier, self).__init__()
        
        self.variant = variant
        self.num_classes = num_classes
        
        # Cargar MobileNetV3 preentrenado
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.mobilenet = models.mobilenet_v3_large(weights=weights)
        else:
            self.mobilenet = models.mobilenet_v3_large(weights=None)
        
        # Congelar capas del modelo base (se pueden descongelar después)
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        
        # Obtener tamaño de entrada del clasificador
        in_features = self.mobilenet.classifier[0].in_features
        
        # Reemplazar el clasificador según la variante
        if variant == 'simple':
            self.mobilenet.classifier = self._build_simple_classifier(in_features, num_classes)
        elif variant == 'extended':
            self.mobilenet.classifier = self._build_extended_classifier(in_features, num_classes)
        else:
            raise ValueError(f"Variante '{variant}' no soportada. Use 'simple' o 'extended'.")
    
    def _build_simple_classifier(self, in_features: int, num_classes: int) -> nn.Sequential:
        """
        Construye un clasificador simple (Versión 1).
        Solo una capa Fully Connected sin Batch Normalization ni Dropout.
        
        Args:
            in_features: Tamaño de entrada
            num_classes: Número de clases
            
        Returns:
            Clasificador simple
        """
        return nn.Sequential(
            nn.Linear(in_features, num_classes)
        )
    
    def _build_extended_classifier(self, in_features: int, num_classes: int) -> nn.Sequential:
        """
        Construye un clasificador extendido tipo embudo (Versión 2).
        Múltiples capas ocultas con Batch Normalization y Dropout.
        
        Args:
            in_features: Tamaño de entrada
            num_classes: Número de clases
            
        Returns:
            Clasificador extendido
        """
        config = Config.ModelVariants.VARIANT_2
        hidden_layers = config['hidden_layers']
        dropout_min, dropout_max = config['dropout_prob_range']
        
        layers = []
        current_features = in_features
        
        # Calcular valores de dropout incrementales
        num_hidden = len(hidden_layers)
        dropout_values = [
            dropout_min + (dropout_max - dropout_min) * i / max(1, num_hidden - 1)
            for i in range(num_hidden)
        ]
        
        # Construir capas ocultas
        for i, hidden_size in enumerate(hidden_layers):
            # Linear
            layers.append(nn.Linear(current_features, hidden_size))
            
            # Batch Normalization
            if config['use_batch_norm']:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activación (ReLU)
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            if config['use_dropout']:
                layers.append(nn.Dropout(p=dropout_values[i]))
            
            current_features = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(current_features, num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada [batch_size, 3, 224, 224]
            
        Returns:
            Logits de salida [batch_size, num_classes]
        """
        return self.mobilenet(x)
    
    def unfreeze_layers(self, num_layers: int = None):
        """
        Descongela capas del modelo base para fine-tuning.
        
        Args:
            num_layers: Número de capas a descongelar desde el final.
                       Si None, descongela todas las capas.
        """
        if num_layers is None:
            # Descongelar todas las capas
            for param in self.mobilenet.parameters():
                param.requires_grad = True
            print("Todas las capas del modelo base han sido descongeladas.")
        else:
            # Descongelar últimas num_layers capas
            layers = list(self.mobilenet.features.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Últimas {num_layers} capas del modelo base han sido descongeladas.")
    
    def freeze_layers(self):
        """Congela todas las capas del modelo base."""
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        print("Todas las capas del modelo base han sido congeladas.")
    
    def get_trainable_params(self) -> Tuple[int, int]:
        """
        Obtiene el número de parámetros entrenables y totales.
        
        Returns:
            Tupla (parámetros entrenables, parámetros totales)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def create_model(variant: str = 'simple', 
                 pretrained: bool = True,
                 num_classes: int = None) -> MobileNetClassifier:
    """
    Función auxiliar para crear un modelo MobileNetClassifier.
    
    Args:
        variant: 'simple' o 'extended'
        pretrained: Si True, usa pesos preentrenados
        num_classes: Número de clases (por defecto usa Config.NUM_CLASSES)
        
    Returns:
        Instancia de MobileNetClassifier
    """
    num_classes = num_classes or Config.NUM_CLASSES
    
    model = MobileNetClassifier(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained
    )
    
    trainable, total = model.get_trainable_params()
    print(f"\nModelo creado: MobileNetV3 - Variante '{variant}'")
    print(f"Parámetros entrenables: {trainable:,}")
    print(f"Parámetros totales: {total:,}")
    print(f"Porcentaje entrenable: {100 * trainable / total:.2f}%")
    
    return model
