# Fase 2 — Computer Vision & Time Series

**Timeline:** Q2 2027 (Abril — Junio)
**Carácter:** Dos verticales de producto con routers propios `/api/v1/cv/*` y `/api/v1/timeseries/*`
**Dependencias:** Fase 0, Fase 1, infraestructura GPU opcional

> 📖 Documentación de referencia: [roadmap.md](../roadmap.md#fase-2--q2-2027-computer-vision--time-series)

---

## Índice

- [Visión General](#visión-general)
- [Computer Vision](#computer-vision-apiv1cv)
  - [2.1 Image Classification](#21-image-classification-fine-tuning)
  - [2.2 Semantic Segmentation](#22-semantic-segmentation)
  - [2.3 Object Detection](#23-object-detection)
  - [2.4 Document Intelligence (OCR)](#24-document-intelligence-ocr--dataframe)
- [Time Series](#time-series-apiv1timeseries)
  - [2.5 Time Series Training](#25-time-series-training)
  - [2.6 Forecasting Inference](#26-time-series-forecasting-inference)
  - [2.7 Anomaly Detection](#27-time-series-anomaly-detection)
- [Stack Tecnológico](#stack-tecnológico)
- [Impacto en la Arquitectura](#impacto-en-la-arquitectura)
- [API Endpoints](#api-endpoints)
- [Riesgos](#riesgos)
- [Plan de Implementación](#plan-de-implementación)

---

## Visión General

Introducir dos verticales nuevas bajo sus propios dominios de ruta:

- **Computer Vision** (`/api/v1/cv/*`): clasificación de imágenes, segmentación semántica, detección de objetos, OCR
- **Time Series** (`/api/v1/timeseries/*`): forecasting y detección de anomalías temporales

Adicionalmente, se integra **Document Intelligence** (OCR → DataFrame) como puente entre documentos escaneados y datos tabulares.

---

## Computer Vision (`/api/v1/cv/*`)

### 2.1 — Image Classification (Fine-tuning)

**Router:** `app/api/routes/v1/cv/training.py`
**Servicio:** `app/services/vision_trainer.py` — `VisionTrainer`

```
POST /api/v1/cv/train                → Entrenar clasificador de imágenes
GET  /api/v1/cv/algorithms           → Listar arquitecturas disponibles
GET  /api/v1/cv/training/status/{id} → Estado del entrenamiento
```

#### Arquitecturas Soportadas (Clasificación)

| API Key | Modelo | Tamaño | Top-1 ImageNet | GPU recomendada |
|---------|--------|--------|----------------|-----------------|
| `mobilenet_v3` | MobileNetV3-Small | ~22 MB | 67.4% | CPU/GPU (muy ligero) |
| `efficientnet_b0` | EfficientNet-B0 | ~20 MB | 77.1% | CPU/GPU |
| `resnet18` | ResNet-18 | ~44 MB | 69.8% | CPU/GPU |
| `resnet50` | ResNet-50 | ~98 MB | 76.0% | GPU recomendada |
| `efficientnet_b3` | EfficientNet-B3 | ~48 MB | 81.1% | GPU |
| `convnext_tiny` | ConvNeXt-Tiny | ~110 MB | 82.0% | GPU |
| `vit_b_16` | Vision Transformer Base | ~330 MB | 81.1% | GPU (necesaria) |

#### Arquitecturas Soportadas (Segmentación)

| API Key | Modelo | Tamaño | mIoU |
|---------|--------|--------|------|
| `lraspp_mobilenet_v3` | LRASPP MobileNetV3 | ~4 MB | 60% |
| `fcn_resnet50` | FCN ResNet-50 | ~130 MB | 66% |
| `deeplabv3_resnet50` | DeepLabV3 ResNet-50 | ~160 MB | 73% |
| `deeplabv3_resnet101` | DeepLabV3 ResNet-101 | ~210 MB | 76% |

#### Dataset Format

- **Clasificación:** ZIP con estructura `ImageFolder` (`class_name/image.jpg`)
- **Segmentación:** ZIP con `images/` + `masks/` (máscaras PNG monocromo)

#### Métricas Trackeadas en MLflow

| Tarea | Métricas |
|-------|----------|
| Clasificación | accuracy, top-5 accuracy, F1, precision, recall, confusion matrix |
| Segmentación | IoU por clase, mean IoU, Dice coefficient, pixel accuracy |

### 2.2 — Semantic Segmentation

**Router:** `app/api/routes/v1/cv/segmentation.py`

```
POST /api/v1/cv/segment                → Segmentar una imagen
POST /api/v1/cv/segment/batch          → Batch segmentation
WS   /api/v1/cv/streaming/{model_id}   → Streaming segmentación en tiempo real
```

Segmentación semántica pixel-wise con modelos DeepLabV3, FCN, o U-Net custom registrados en `ModelFactory`.

**Integración con incertidumbre existente:** Los estimadores MC Dropout, TTA y ensemble ya soportan outputs `(B, C, H, W)` — funcionan directamente con modelos de segmentación. El mapa de incertidumbre se solapa sobre la máscara de segmentación en la UI.

#### Output

```json
{
  "mask": "base64_encoded_png",
  "overlay": "base64_encoded_png",
  "class_scores": {"background": 0.02, "car": 0.95, "pedestrian": 0.03},
  "uncertainty_map": "base64_encoded_png",
  "mean_iou": 0.87,
  "latency_ms": 45.2
}
```

### 2.3 — Object Detection

**Router:** `app/api/routes/v1/cv/detection.py`

```
POST /api/v1/cv/detect                → Detectar objetos en imagen
POST /api/v1/cv/detect/batch          → Batch detection
```

Integra modelos YOLO (`ultralytics` YOLOv8/v11) y modelos custom TorchScript.

#### Output

```json
{
  "detections": [
    {"bbox": [x1, y1, x2, y2], "class": "car", "confidence": 0.95},
    {"bbox": [x1, y1, x2, y2], "class": "pedestrian", "confidence": 0.87}
  ],
  "image_with_boxes": "base64_encoded_jpg",
  "latency_ms": 32.1
}
```

### 2.4 — Document Intelligence (OCR → DataFrame)

Post-hook en `POST /api/v1/datasets/`: acepta PDFs/imágenes escaneadas y los convierte automáticamente a DataFrames tabulares.

**Tecnología:** `docTR` (Apache 2.0) — layout analysis + OCR.
**Pipeline:**
1. Detección de layout (tablas, párrafos, títulos)
2. OCR por región
3. Estructuración en DataFrame
4. Almacenamiento como dataset tabular estándar

---

## Time Series (`/api/v1/timeseries/*`)

### 2.5 — Time Series Training

**Router:** `app/api/routes/v1/timeseries/training.py`
**Servicio:** `app/services/timeseries_trainer.py` — `TimeSeriesTrainer`

```
POST /api/v1/timeseries/train            → Entrenar modelo de forecasting
GET  /api/v1/timeseries/algorithms       → Listar algoritmos disponibles
GET  /api/v1/timeseries/training/status/{id} → Estado del entrenamiento
```

#### Algoritmos Soportados

| API Key | Categoría | Librería | Soporta exógenas | Intervalos confianza |
|---------|-----------|----------|-----------------|---------------------|
| `arima` | Clásico | `statsforecast` | ❌ | ✅ (conformal) |
| `ets` | Clásico | `statsforecast` | ❌ | ✅ |
| `theta` | Clásico | `statsforecast` | ❌ | ✅ |
| `ces` | Clásico | `statsforecast` | ❌ | ✅ |
| `lightgbm_ts` | ML | `mlforecast` | ✅ | ❌ (quantile) |
| `xgboost_ts` | ML | `mlforecast` | ✅ | ❌ (quantile) |
| `nbeats` | Deep Learning | `neuralforecast` | ❌ | ✅ |
| `nhits` | Deep Learning | `neuralforecast` | ❌ | ✅ |
| `tft` | Deep Learning | `neuralforecast` | ✅ | ✅ |
| `patchtst` | Deep Learning | `neuralforecast` | ❌ | ✅ |

#### Dataset Format

CSV con al menos:
- Columna temporal (`datetime`)
- Columna target numérica
- Opcional: columnas exógenas (numéricas)

#### Métricas

MAE, RMSE, MAPE, SMAPE, MASE

### 2.6 — Time Series Forecasting (Inference)

**Router:** `app/api/routes/v1/timeseries/forecast.py`

```
POST /api/v1/timeseries/forecast           → Generar pronóstico N pasos adelante
POST /api/v1/timeseries/forecast/streaming → Streaming forecast (rolling window)
GET  /api/v1/timeseries/forecast/{id}      → Obtener predicción con intervalos
```

Output enriquecido:
- Predicción puntual
- **Intervalos de confianza** (conformal prediction o quantile regression)
- Componentes de la serie (tendencia, estacionalidad, residual)

```json
{
  "forecast": [100.5, 101.2, 102.0, ...],
  "confidence_intervals": {
    "lower": [98.1, 98.5, 99.0, ...],
    "upper": [102.9, 103.9, 105.0, ...]
  },
  "components": {
    "trend": [99.0, 99.5, 100.0, ...],
    "seasonal": [1.5, 1.7, 2.0, ...]
  },
  "metrics": {"mape": 3.2, "mase": 0.85}
}
```

### 2.7 — Time Series Anomaly Detection

**Router:** `app/api/routes/v1/timeseries/anomalies.py`

```
POST /api/v1/timeseries/anomalies          → Detectar anomalías en serie temporal
PATCH /api/v1/timeseries/anomalies/config  → Configurar sensibilidad
```

Detecta puntos anómalos via:
- Residuos del modelo de forecasting + thresholds Z-score / IQR
- Modelos dedicados: Isolation Forest, Autoencoder (PyOD)

---

## Stack Tecnológico

| Componente | Tecnología | Justificación |
|-----------|-----------|---------------|
| **Image Classification** | `torchvision.models` (pretrained) + fine-tuning | Ya tenemos PyTorch. Incluye ViT, ConvNeXt, EfficientNet |
| **Segmentation** | `torchvision.models.segmentation` (DeepLabV3, FCN) | Modelos pre-entrenados integrados en torchvision |
| **Object Detection** | `ultralytics` (YOLOv8/v11) | API limpia, TorchScript export, MLflow compatible |
| **Image Augmentation** | `albumentations` | Augmentation para clasificación + segmentación (masks) |
| **OCR** | `docTR` (Apache 2.0) | Layout analysis + OCR nativo. Modelos ~200 MB |
| **TS Clásica** | `statsforecast` (Nixtla) | ARIMA/ETS/Theta ultrarrápidos. 100x más rápido que `statsmodels` |
| **TS ML** | `mlforecast` (Nixtla) | LightGBM/XGBoost con feature engineering temporal automático |
| **TS Deep Learning** | `neuralforecast` (Nixtla) | N-BEATS, N-HiTS, TFT, PatchTST — state-of-the-art |
| **TS Plotting** | `utilsforecast` | Visualización de forecasts con intervalos de confianza |

---

## Impacto en la Arquitectura

| Cambio | Detalle |
|--------|---------|
| **Nuevo paquete CV** | `app/api/routes/v1/cv/` — training, segmentation, detection |
| **Nuevo paquete TS** | `app/api/routes/v1/timeseries/` — training, forecast, anomalies |
| **Nuevo servicio** | `app/services/vision_trainer.py` — Clasificación + Segmentación |
| **Nuevo servicio** | `app/services/timeseries_trainer.py` — Forecasting + anomalías |
| **Nuevo servicio** | `app/services/document_intelligence.py` — OCR + tablas |
| **Extensión ORM `Dataset`** | Nuevo campo `data_type: str` (values: `tabular`, `image`, `timeseries`, `document`) |
| **Celery tasks** | `train_vision.py`, `train_timeseries.py` |
| **Frontend** | `SegmentationViewer.tsx`, `DetectionViewer.tsx`, `ForecastChart.tsx` |
| **`hyperparams.py`** | Nuevas secciones `CV_ALGORITHMS`, `TIMESERIES_ALGORITHMS` |

---

## API Endpoints

### Computer Vision

| Método | Path | Descripción |
|--------|------|-------------|
| POST | `/api/v1/cv/train` | Entrenar clasificador/segmentador |
| GET | `/api/v1/cv/algorithms` | Listar arquitecturas disponibles |
| GET | `/api/v1/cv/training/status/{id}` | Estado del entrenamiento |
| POST | `/api/v1/cv/segment` | Segmentar una imagen |
| POST | `/api/v1/cv/segment/batch` | Batch segmentation |
| WS | `/api/v1/cv/streaming/{model_id}` | Streaming segmentación |
| POST | `/api/v1/cv/detect` | Detectar objetos en imagen |
| POST | `/api/v1/cv/detect/batch` | Batch detection |

### Time Series

| Método | Path | Descripción |
|--------|------|-------------|
| POST | `/api/v1/timeseries/train` | Entrenar modelo de forecasting |
| GET | `/api/v1/timeseries/algorithms` | Listar algoritmos disponibles |
| GET | `/api/v1/timeseries/training/status/{id}` | Estado del entrenamiento |
| POST | `/api/v1/timeseries/forecast` | Generar pronóstico |
| POST | `/api/v1/timeseries/forecast/streaming` | Streaming forecast |
| GET | `/api/v1/timeseries/forecast/{id}` | Obtener predicción |
| POST | `/api/v1/timeseries/anomalies` | Detectar anomalías |
| PATCH | `/api/v1/timeseries/anomalies/config` | Configurar sensibilidad |

---

## Riesgos

| Riesgo | Mitigación |
|--------|-----------|
| **GPU para CV** | Segmentación y clasificación requieren GPU para fine-tuning. Ofrecer modelos ligeros (MobileNet, EfficientNet-B0) como fallback CPU |
| **Tamaño de modelos** | DeepLabV3 ~200 MB, YOLO ~40 MB. `ModelCache` existente con LRU eviction |
| **Datasets de imagen pesados** | Quotas por `file_size_bytes`. Thumbnails automáticos en MinIO para UI |
| **Frecuencia en TS** | El usuario debe especificar frecuencia (H, D, W, M). Auto-detección con `pd.infer_freq` como fallback |
| **Nixtla ecosystem** | statsforecast + mlforecast + neuralforecast comparten API similar. Si una falla, las otras son drop-in replacements |
| **Calidad de OCR** | docTR funciona bien con documentos impresos. Para manuscritos, considerar modelos específicos |

---

## Plan de Implementación

| Sub-fase | Contenido | Esfuerzo | Dependencias |
|----------|-----------|----------|-------------|
| **2.1** | VisionTrainer (clasificación) + endpoints + MLflow | 🔴 3 sem | GPU disponible |
| **2.2** | Segmentación + incertidumbre + endpoints | 🔴 3 sem | 2.1 (parcial) |
| **2.3** | YOLO detection + endpoints | 🟡 2 sem | 2.1 (débil) |
| **2.4** | Document Intelligence (OCR) | 🟡 2 sem | Ninguna |
| **2.5** | TimeSeriesTrainer + algoritmos clásicos + endpoints | 🟡 2-3 sem | Ninguna |
| **2.6** | TS Forecasting inference + intervalos confianza + streaming | 🟡 2 sem | 2.5 |
| **2.7** | TS Anomaly Detection + endpoints | 🟡 1-2 sem | 2.5 |

---

> [← Volver al Roadmap principal](../roadmap.md)
