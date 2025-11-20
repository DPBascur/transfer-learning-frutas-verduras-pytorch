"""
Módulo de entrenamiento del modelo con seguimiento de métricas y checkpoints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime

from src.training.early_stopping import EarlyStopping


class Trainer:
    """
    Clase para entrenar modelos de clasificación con seguimiento de métricas.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str = 'cuda',
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 early_stopping: Optional[EarlyStopping] = None,
                 save_dir: Path = None):
        """
        Args:
            model: Modelo a entrenar
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            criterion: Función de pérdida
            optimizer: Optimizador
            device: Dispositivo ('cuda' o 'cpu')
            scheduler: Scheduler de learning rate (opcional)
            early_stopping: Objeto EarlyStopping (opcional)
            save_dir: Directorio para guardar checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.save_dir = save_dir
        
        # Historial de métricas
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Entrena el modelo por una epoch.
        
        Returns:
            Tupla (pérdida promedio, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Métricas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progreso
            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch [{batch_idx + 1}/{len(self.train_loader)}] | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {100. * correct / total:.2f}%', end='\r')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """
        Valida el modelo.
        
        Returns:
            Tupla (pérdida promedio, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs: int, model_name: str = 'model') -> Dict[str, List[float]]:
        """
        Entrena el modelo por múltiples epochs.
        
        Args:
            num_epochs: Número de epochs
            model_name: Nombre base para guardar checkpoints
            
        Returns:
            Historial de métricas
        """
        print(f"\nIniciando entrenamiento por {num_epochs} epochs...")
        print(f"Dispositivo: {self.device}")
        print(f"Tamaño dataset entrenamiento: {len(self.train_loader.dataset)}")
        print(f"Tamaño dataset validación: {len(self.val_loader.dataset)}")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Entrenamiento
            train_loss, train_acc = self.train_epoch()
            
            # Validación
            val_loss, val_acc = self.validate()
            
            # Learning rate actual
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Actualizar scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Guardar métricas
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Tiempo de epoch
            epoch_time = time.time() - epoch_start
            
            # Imprimir resultados
            print(f"\nEpoch [{epoch}/{num_epochs}] | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Guardar mejor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.save_dir:
                    self.best_model_path = self.save_checkpoint(
                        epoch, model_name, is_best=True
                    )
                    print(f"  Nuevo mejor modelo guardado! Acc: {val_acc:.2f}%")
            
            # Early Stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, epoch):
                    print(f"\nEntrenamiento detenido por Early Stopping en epoch {epoch}")
                    break
            
            print("-" * 80)
        
        total_time = time.time() - start_time
        print(f"\nEntrenamiento completado en {total_time / 60:.2f} minutos")
        print(f"Mejor accuracy de validación: {self.best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, epoch: int, model_name: str, is_best: bool = False) -> Path:
        """
        Guarda un checkpoint del modelo.
        
        Args:
            epoch: Epoch actual
            model_name: Nombre base del modelo
            is_best: Si True, guarda como mejor modelo
            
        Returns:
            Ruta del archivo guardado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            filename = f"{model_name}_best.pth"
        else:
            filename = f"{model_name}_epoch_{epoch}_{timestamp}.pth"
        
        filepath = self.save_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        return filepath
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Carga un checkpoint del modelo.
        
        Args:
            checkpoint_path: Ruta al archivo del checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint cargado desde: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Best Val Acc: {self.best_val_acc:.2f}%")
