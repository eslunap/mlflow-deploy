# Taller 4 – Pipeline CI/CD con GitHub Actions y MLflow

## Descripción

Este proyecto implementa un pipeline de CI/CD para Machine Learning usando GitHub Actions y MLflow. El flujo automatiza el entrenamiento, registro y validación de un modelo de aprendizaje automático, ejecutándose automáticamente al realizar un `push` sobre la rama `main`.

## Estructura del proyecto

```text
mlflow-deploy/
├── train.py
├── validate.py
├── requirements.txt
├── Makefile
├── README.md
├── data/
│   └── winequality-red.csv
├── mlruns/
└── .github/
    └── workflows/
        └── mlflow-ci.yml

```        

## Dataset utilizado

Se utilizó el dataset externo Wine Quality Red en formato CSV. Este dataset contiene variables fisicoquímicas de muestras de vino tinto y una variable objetivo denominada `quality`.

Se seleccionó este dataset porque permite construir un problema de regresión sin depender de `sklearn.datasets`, cumpliendo con el requisito de usar una fuente externa.

## Entrenamiento y registro del modelo

El archivo `train.py` realiza las siguientes acciones:

- Carga el dataset externo desde la carpeta `data/`.
- Separa las variables predictoras y la variable objetivo.
- Entrena un modelo `RandomForestRegressor`.
- Calcula métricas de desempeño como `MSE` y `R2`.
- Registra en MLflow:
    - Parámetros del modelo.
    - Métricas de desempeño.
    - Modelo entrenado.
    - Firma del modelo.
    - Input example.
    - Artefactos generados.
    - Validación del modelo

El archivo `validate.py` carga el modelo registrado desde MLflow usando el run_id generado durante el entrenamiento. Luego evalúa el modelo sobre datos de prueba y valida que el MSE cumpla con el umbral definido.

Si el modelo no cumple el umbral, el pipeline falla automáticamente.

## Métricas obtenidas

Durante la ejecución local del entrenamiento se obtuvieron los siguientes resultados:

    Entrenamiento finalizado correctamente.
    MSE: 0.3012
    R2: 0.5390

Las métricas registradas fueron:

| Métrica |  Valor |
| ------- | -----: |
| MSE     | 0.3012 |
| R2      | 0.5390 |

## Validación del modelo
El archivo `validate.py` carga el modelo registrado desde MLflow usando el `run_id` generado durante el entrenamiento.

Posteriormente, evalúa el modelo sobre los datos de prueba y valida que el valor de MSE sea menor o igual al umbral definido.

Para este ejercicio se definió el siguiente umbral:

```MSE_THRESHOLD = 0.5```

Resultado de la validación local:


```
Modelo cargado desde MLflow: runs:/6f0275b005b542d596f98d0e48a63f48/model
MSE obtenido: 0.3012
R2 obtenido: 0.5390
Umbral máximo permitido de MSE: 0.5
Validación exitosa: el modelo cumple el umbral definido.
```
Esto indica que el modelo cumple con el criterio mínimo de desempeño establecido para continuar el pipeline.

## Uso del Makefile

El proyecto incluye un `Makefile` para simplificar la ejecución de comandos.

Comandos disponibles:

    make install
    make train
    make validate
    make all

Descripción de los comandos:

| Comando         | Descripción                                                |
| --------------- | ---------------------------------------------------------- |
| `make install`  | Instala las dependencias definidas en `requirements.txt`.  |
| `make train`    | Ejecuta el entrenamiento y registro del modelo con MLflow. |
| `make validate` | Ejecuta la validación automática del modelo registrado.    |
| `make all`      | Ejecuta instalación, entrenamiento y validación.           |

## Pipeline con GitHub Actions

Se creó el workflow:

    .github/workflows/mlflow-ci.yml

Este workflow automatiza el proceso de CI/CD para el proyecto.

El pipeline realiza los siguientes pasos:

1. Clona el repositorio.
2. Instala `make`.
3. Configura Python.
4. Instala las dependencias del proyecto.
5. Ejecuta el entrenamiento del modelo.
6. Registra el modelo, métricas y artefactos en MLflow.
7. Ejecuta la validación automática del modelo.
8. Publica los artefactos generados por el workflow.

## Artefactos generados

Durante la ejecución del pipeline en GitHub Actions se generó el artefacto:

    mlflow-tracking-artifacts

Este artefacto contiene los elementos producidos por el pipeline, incluyendo:

- Registros de MLflow.
- Modelo entrenado.
- Métricas calculadas.
- Archivo `run_id.txt`.

## Resultado del pipeline

El workflow de GitHub Actions finalizó correctamente en estado **Success**.

Esto evidencia que el pipeline pudo ejecutarse de forma automática, entrenando el modelo, registrando resultados con MLflow, validando el desempeño y generando artefactos descargables.

## Consideraciones técnicas

Durante el desarrollo se excluyeron del control de versiones las carpetas generadas localmente, como:

    mlruns/
    model/
    metrics/
    run_id.txt
    venv/

Esto se hizo para evitar conflictos entre rutas locales de Windows y el entorno Linux utilizado por GitHub Actions.

Los artefactos se generan directamente durante la ejecución del workflow y se publican como artefactos descargables desde GitHub Actions.

## Evidencias del taller

Las evidencias principales del desarrollo son:

- Repositorio con la estructura solicitada.
- Archivo `train.py` funcional.
- Archivo `validate.py` funcional.
- Archivo `Makefile` con comandos de ejecución.
- Workflow `mlflow-ci.yml` configurado.
- Ejecución exitosa del pipeline en GitHub Actions.
- Artefacto `mlflow-tracking-artifacts` generado.
- Validación exitosa del modelo con `MSE = 0.3012`.

## Ejecución local

Para ejecutar el proyecto localmente:

    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    python train.py
    python validate.py

## Ejecución mediante Makefile

En entornos donde `make` esté disponible:

    make install
    make train
    make validate
    make all
