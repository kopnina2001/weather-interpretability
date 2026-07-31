# Weather Model Interpretability

Исследование методов интерпретируемости для transformer-based AI-моделей погоды на кластере "Летний кластер AIRI". Кейс: тайфун Bebinca (Китай, сентябрь 2024) и спокойный контроль (ноябрь 2024).

## Кластер

- 4× NVIDIA Tesla V100-SXM3-32GB (32GB VRAM каждая)
- 16 vCPU Intel Xeon (Skylake, IBRS)
- 251 GiB RAM, 2.1TB свободного диска
- Ubuntu 22.04.5 LTS, kernel 5.15
- CDS API уже настроен (`~/.cdsapirc`) — доступен прямой доступ к ERA5 через Copernicus CDS, в дополнение к ARCO ERA5

## Модели

| Модель | Архитектура | Разрешение | Формат | Статус в проекте |
|---|---|---|---|---|
| Pangu-Weather | 3D Earth-Specific Transformer | 0.25°, 13 pressure levels | ONNX | verification campaign (96 init), occlusion saliency (notebook 02) |
| Aurora | Perceiver-encoder + 3D Swin Transformer U-Net backbone | 0.25° | PyTorch | полный набор экспериментов: campaign, error maps, input patching, skeleton/tensor decomposition |
| Stormer | плоский ViT (24 identical transformer blocks) | 1.40625° (грубая), нормализованное пространство | PyTorch | verification campaign, skeleton decomposition, input patching |
| GraphCast | GNN | — | — | вне scope — не transformer-архитектура |

## Входные поля

- Поверхность: MSLP, 10m U/V wind, 2m temperature
- Верхние уровни: geopotential, specific humidity, temperature, u/v wind на 13 уровнях давления: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 гПа
- Статика (Aurora): lsm, slt, z (орография)

## Источники данных

**ARCO ERA5** (сырые поля, ground truth):
```python
import xarray as xr
ds = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    storage_options=dict(token='anon'))
```
Кэш на 2 года (2024-2025, 6-часовой) — `/srv/exw/data/irina_weather_interpretability/era5_2024_2025_6h_{surface,upper}.zarr`.

**WeatherBench2 climatology** (precomputed, day-of-year × hour-of-day, 1990-2019):
```python
ds = xr.open_zarr('gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr',
                   storage_options=dict(token='anon'))
```
Кэш — `climatology_1990-2019_{surface,upper_500_850_1000}.zarr`. **Важно**: climatology для upper-air переменных скачана только на 3 уровня (500/850/1000 hPa) из 13 — любой эксперимент, патчащий upper-air переменную климатологией, patch-ит только эти 3 уровня (partial-column), остальные 10 остаются реальными данными. Это задокументировано в каждом релевантном скрипте/ноутбуке.

`chunks=None` в `xr.open_zarr()` — критично для избежания dask-related OOM при работе с большими срезами.

## Реализованные методы и находки

### 1. Верификационная кампания (notebooks 01-06)
96 инициализаций (2024-2025, 4/месяц) × 3 модели × 8 lead times (6-48h). Метрики: RMSE, ACC (precise WB2 climatology), bias, variance ratio, corr-with-climatology, zonal power spectrum. Прогнозы кэшированы (`results/campaign_*_predictions/`), метрики пересчитываются без перезапуска моделей.

### 2. Error maps для конкретного кейса (notebook 06, script `plot_china_error_maps.py`)
Тайфун Bebinca, zoom на Китай: MSLP + Z@850hPa error maps (pred−truth), включая climatology-mean baseline для сравнения (модели ~5x лучше naive climatology в зоне шторма).

### 3. Skeleton (interpolative) decomposition — notebook 07
Вместо SVD/PCA (абстрактные eigenvectors) — `scipy.linalg.interpolative.interp_decomp`: выбирает k **реальных** точек сетки (landmarks) / каналов, через которые восстанавливается всё hidden state. Применено к backbone Aurora (3 encoder stage) и Stormer (24 flat ViT blocks, hook на shallow/middle/deep).

**Главная находка**: landmarks кластеризуются на центре тайфуна (850hPa vorticity extremum) — подтверждено quiet-baseline контролем (15 ноября, без шторма — landmarks не кластеризуются на сопоставимом экстремуме). Воспроизведено на **двух архитектурно разных моделях** (Aurora Swin U-Net, Stormer flat ViT).

Попытка того же метода на attention-матрицах (вместо hidden state) дала **negative result**: landmarks садятся на края/углы окна независимо от данных — артефакт rank-revealing QR, подтверждено тем же quiet-контролем.

### 4. Input-level climatological patching — notebook 08 (+ `_stormer` вариант)
Причинная абляция: заменить ОДНУ input-компоненту (surface_mslp / wind / mass_field / temperature / t2m) на climatology mean, оставить остальное реальным, прогнать полный rollout, сравнить с baseline. Два scope (global/regional), 4 метрики (RMSE vs baseline, ΔRMSE vs truth, ΔACC, relative sensitivity) + 2 взвешенные (area-weighted, storm-focused Gaussian на MSLP-min track).

**Находка**: wind/mass_field патчи сильнее всего портят Z-поля (mass-wind balance); MSLP имеет непропорционально широкое кросс-влияние для одной 2D-переменной.

### 5. Matrix/tensor decomposition для redundancy heads/layers — notebook 09
- **Spectrum/effective rank** (SVD hidden state) — negative result: storm/quiet почти идентичны, сжимаемость структурна, не event-driven.
- **CKA** между attention heads — positive: heads менее избыточны/более специализированы при реальном шторме.
- **Tucker decomposition** (heads, query, key) тензора — независимо подтверждает CKA строгим численным методом.
- **RepE-style PCA** на разнице storm−quiet hidden state — negative result: главная компонента оказалась сезонным confound (Sept vs Nov), не сигналом шторма — методологический урок (нужны contrastive пары в одном сезоне, или reuse patching-инфраструктуры вместо двух разных дат).
- **NMF** на attention-весах — чище структура, чем skeleton decomposition на том же объекте, но архитектурное свойство (одинаково у storm/quiet), не storm-specific.

## Ноутбуки

- `01_era5_visualization.ipynb` — визуализация синоптических кейсов из ARCO ERA5
- `02_saliency_experiment.ipynb` — occlusion saliency на Pangu-Weather
- `03_output_correlation.ipynb` — корреляционная матрица выходов модели
- `04_aurora_vs_pangu.ipynb` — baseline-сравнение Aurora vs Pangu-Weather
- `05_model_comparison.ipynb` — сравнение всех трёх моделей
- `06_climatology_correlation.ipynb` — verification campaign метрики (RMSE/ACC/bias/variance ratio/spectral), error maps для тайфуна Bebinca
- `07_skeleton_decomposition.ipynb` — skeleton decomposition hidden state (Aurora + Stormer), attention negative result, quiet-baseline контроль
- `08_input_patching.ipynb` / `08_input_patching_stormer.ipynb` — causal input-level climatological patching
- `09_tensor_decomposition.ipynb` — spectrum/effective rank, CKA, Tucker, RepE-PCA, NMF

## Aurora — важные находки (см. `run_case_aurora.py`)

Три нюанса, из-за которых прямое использование `aurora.rollout()` не работало на одной V100 (32GB):

1. **`Metadata.time` — один элемент, не по одному на каждый входной таймстеп.** Aurora требует 2 входных
   времени (t-6h и t) в тензорах, но `metadata.time` — всегда 1-кортеж (текущее/референсное время). Передача
   2-элементного кортежа не бросает ошибку, но задирает потребление памяти на многошаговом rollout до OOM —
   баг тихий, трудно диагностируемый.
2. **`aurora.rollout()` сама по себе даёт лишний memory overhead** (~8GB) относительно прямого вызова
   `model.forward()` — вероятно, из-за повторного `batch_transform_hook`/`type`/`crop`/`to(device)` на каждом
   шаге. Обошли: ручной 4-шаговый rollout, вызывая `model.forward()` напрямую и продвигая batch через
   `aurora.rollout._advance_batch()` (тот же helper, что использует сама `rollout()`).
3. **Выходная сетка — 720 широт, не 721** (модель обрезает южный полюс, -90°). Нужно один раз обрезать входной
   batch через `batch.crop(model.patch_size)` перед циклом, иначе `_advance_batch` падает на несовпадении shape.

С `torch.autocast('cuda', dtype=torch.float16)` (не `model.half()` — ломает числовую проверку на `lat`/`lon`,
которые должны остаться fp32) — стабильно **19.6GB на шаг**, 4 шага (24h) укладываются с большим запасом.

## Окружение

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

**Важно для GPU**: `onnxruntime-gpu` не находит CUDA-библиотеки сам по себе — нужно указать `LD_LIBRARY_PATH` на
CUDA-либы, которые уже устанавливает `torch` (отдельный system CUDA toolkit не нужен). Добавить в конец
`~/venv/bin/activate`:

```bash
export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -):$LD_LIBRARY_PATH
```

(env-переменная, установленная уже ПОСЛЕ старта Python-процесса, не подхватывается — это должно быть именно в
`activate`, а не в коде ноутбука после `import onnxruntime`.)

## Совместный доступ (shared filesystem `/srv/exw`)

Тяжёлые артефакты (веса модели, входные данные, результаты прогонов) не лежат в git — они в общей папке `/srv/exw`,
следуя существующей конвенции команды (`<username>_<project>` для `data/`/`runs/`, без префикса для переиспользуемых
весов в `checkpoints/`):

- `/srv/exw/checkpoints/pangu_weather_{6,24}/pangu_weather_{6,24}.onnx` — веса модели (общие, не per-project)
- `/srv/exw/data/irina_weather_interpretability/` — входные данные (ERA5 2024-2025, WB2 climatology)
- Результаты прогонов (`results/*`) — в `.gitignore`, живут только на диске (не в git), пересчитываются скриптами при необходимости

Симлинки `data` и `results` закоммичены в git (git symlink, mode 120000) — восстанавливаются автоматически при
`git clone`. Только `model_weights/` — сама папка в `.gitignore`, её нужно собрать руками один раз:

```bash
cd weather-interpretability
mkdir -p model_weights
ln -s /srv/exw/checkpoints/pangu_weather_6/pangu_weather_6.onnx model_weights/pangu_weather_6.onnx
ln -s /srv/exw/checkpoints/pangu_weather_24/pangu_weather_24.onnx model_weights/pangu_weather_24.onnx
```

## Workflow: ветки

**В `main` пишет только владелец репозитория.** Остальные — через собственную ветку и Pull Request:

```bash
git checkout -b <имя>/<фича>
# ... изменения, коммиты ...
git push -u origin <имя>/<фича>
# затем открыть PR в main на GitHub
```

## Ссылки

- Pangu-Weather: https://github.com/198808xc/Pangu-Weather
- Aurora: https://github.com/microsoft/aurora
- Stormer: https://github.com/tung-nd/stormer
- ARCO ERA5: https://github.com/google-research/arco-era5
- WeatherBench2: https://github.com/google-research/weatherbench2
- Advection Heads in an Atmosphere Foundation Model (attention analysis в Aurora, референс для notebook 07/09): https://arxiv.org/abs/2508.00969
- Representation Engineering (RepE): https://arxiv.org/abs/2310.01405
- CKA (Kornblith et al. 2019, Similarity of Neural Network Representations Revisited): https://arxiv.org/abs/1905.00414
- DEIM (Chaturantabut & Sørensen 2010) — sensor-placement предок skeleton-decomposition подхода
- Interpretable ML for Weather and Climate Prediction (Survey): https://arxiv.org/pdf/2403.18864

## Статус

Все 9 ноутбуков реализованы. Verification campaign завершена (96 init × 3 модели). Skeleton decomposition, input
patching и tensor/matrix decomposition эксперименты выполнены для Aurora, input patching и skeleton decomposition
— также для Stormer (независимое подтверждение находки про центр шторма на второй архитектуре). Pangu-Weather пока
только в verification campaign и occlusion saliency — input patching и hidden-state интерпретация для Pangu (ONNX,
нет прямого доступа к промежуточным активациям) остаются as future work.

Честные negative results задокументированы наравне с позитивными (attention-skeleton edge-artifact, RepE seasonal
confound, NMF архитектурное а не storm-specific свойство) — часть научной строгости методологии, не пробелы.
