# Scripts CLI

Esta carpeta contiene scripts de línea de comandos para operaciones avanzadas del proyecto.

## Archivos

### `quick_start.py`
Script de verificación de instalación y configuración del entorno.

```bash
python scripts/quick_start.py
```

Verifica:
- Instalación de dependencias
- Disponibilidad de GPU
- Estructura de directorios
- Acceso al dataset

### `train.py`
Script completo de entrenamiento con opciones configurables.

```bash
# Ejemplo: Entrenar versión simple
python scripts/train.py --variant simple --epochs 50 --batch-size 32 --lr 0.001

# Ejemplo: Entrenar versión extendida
python scripts/train.py --variant extended --epochs 50 --batch-size 32
```

**Argumentos:**
- `--variant`: `simple` o `extended` (default: `simple`)
- `--epochs`: Número de epochs (default: 50)
- `--batch-size`: Tamaño del batch (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--device`: `cuda` o `cpu` (default: `cuda`)
- `--no-early-stopping`: Deshabilitar early stopping

### `evaluate.py`
Script de evaluación de modelos guardados.

```bash
python scripts/evaluate.py --model-path saved_models/mobilenet_v3_simple_best.pth --variant simple --dataset test
```

**Argumentos:**
- `--model-path`: Ruta al modelo guardado (requerido)
- `--variant`: `simple` o `extended` (default: `simple`)
- `--batch-size`: Tamaño del batch (default: 32)
- `--device`: `cuda` o `cpu` (default: `cuda`)
- `--dataset`: `train`, `val` o `test` (default: `test`)

### `compare.py`
Script para comparar dos variantes de modelos y generar reportes.

```bash
python scripts/compare.py --model-v1 saved_models/mobilenet_v3_simple_best.pth --model-v2 saved_models/mobilenet_v3_extended_best.pth
```

**Argumentos:**
- `--model-v1`: Ruta al modelo Versión 1 (requerido)
- `--model-v2`: Ruta al modelo Versión 2 (requerido)
- `--experiment-name`: Nombre del experimento para el reporte (default: `comparison`)

Genera:
- Comparación de métricas
- Matrices de confusión
- Gráficos comparativos
- Reporte en formato JSON

## Nota

Para la mayoría de usuarios, se recomienda usar la **interfaz web de Streamlit** (`streamlit run app/app.py`) que ofrece todas estas funcionalidades de forma más intuitiva.

Estos scripts CLI son útiles para:
- Automatización de experimentos
- Integración en pipelines CI/CD
- Ejecución en servidores remotos sin interfaz gráfica
- Scripts de entrenamiento batch
