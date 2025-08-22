# SAR Vegetation and Canopy Analysis

Este repositorio contiene el flujo completo de procesamiento y modelado de parámetros de vegetación
utilizando datos SAR Sentinel-1 y datos de campo en un cultivo de cítricos. El proyecto está estructurado
para separar la **vegetación baja** (pastos y cultivos jóvenes/maduros) y el **dosel/arbolado**.

## Estructura del repositorio

- `data/` : Datos brutos, procesados y externos.
- `notebooks/` : Análisis exploratorios y pruebas de modelos.
- `src/` : Código fuente modular para:
  - Preparación de datos (`data/`)
  - Selección de variables y cálculo de índices (`features/`)
  - Modelos paramétricos y de ML (`models/`)
  - Funciones auxiliares (`utils/`)
- `results/` : Resultados generados por cada análisis, separados por vegetación y dosel.
- `tests/` : Tests unitarios e integraciones para asegurar reproducibilidad.

## Entorno

Se recomienda crear un entorno Conda con:

```bash
conda env create -f environment.yml
conda activate sar_vegetation
```

Si el `yml`se actualiza, utiliza:
```bash
conda env update -f environment.yml --prune
```

# Inicialización
Para poder usar `src/` como un paquete Python instalable en modo editable para importar cualquier módulo dentro de `src/`, debes activar el entorno, elegir el directorio de trabajo apropiado e instalar en modo editable:

```bash
cd route/to/SarVeg
conda activate sar_vegetation
pip install -e .
```
> Aquí el `.` significa “el paquete definido en el setup.py en esta carpeta”. Esto instalará todas las subcarpetas dentro de `src/` como un paquete editable.

# Flujo de trabajo general

1. **Preparación de datos**: cargar las imágenes SAR y los datos de campo, recortar las regiones de interés y generar los datasets procesados.

2. **Selección de variables**: realizar análisis PCA, PLS, y selección por importancia (Random Forest o LightGBM).

3. **Modelado**:

- Regresión lineal paramétrica

- Nadaraya–Watson (no paramétrica)

- Random Forest

- LightGBM

4. **Evaluación y resultados**: métricas (RMSE, MAE, R², rRMSE), gráficos de residuales, predicciones vs. observado.