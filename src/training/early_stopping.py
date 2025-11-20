"""
Implementación de Early Stopping para prevenir overfitting.
"""

import numpy as np


class EarlyStopping:
    """
    Early Stopping para detener el entrenamiento cuando la métrica de validación
    no mejora después de un número determinado de epochs.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Número de epochs a esperar sin mejora
            min_delta: Mejora mínima requerida para considerar progreso
            mode: 'min' para métricas que deben minimizarse (loss),
                  'max' para métricas que deben maximizarse (accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode '{mode}' no soportado. Use 'min' o 'max'.")
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Verifica si se debe detener el entrenamiento.
        
        Args:
            score: Valor de la métrica actual
            epoch: Epoch actual
            
        Returns:
            True si se debe detener el entrenamiento
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly Stopping activado. Sin mejora en {self.patience} epochs.")
                print(f"Mejor epoch: {self.best_epoch} con score: {self.best_score:.4f}")
        
        return self.early_stop
    
    def reset(self):
        """Reinicia el estado del Early Stopping."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
