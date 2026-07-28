# Weather Model Interpretability

Исследование методов интерпретируемости для transformer-based AI-моделей погоды на кластере "Летний кластер AIRI".

## Кластер

- 4× NVIDIA Tesla V100-SXM3-32GB (32GB VRAM каждая)
- 16 vCPU Intel Xeon (Skylake, IBRS)
- 251 GiB RAM, 2.1TB свободного диска
- Ubuntu 22.04.5 LTS, kernel 5.15
- CDS API уже настроен (`~/.cdsapirc`) — доступен прямой доступ к ERA5 через Copernicus CDS, в дополнение к ARCO ERA5

## Модели-кандидаты (transformer architecture)

| Модель | Архитектура | VRAM (инференс) | Вердикт для кластера |
|---|---|---|---|
| Pangu-Weather | 3D Earth-Specific Transformer (ONNX) | ~16–24GB | подходит, одна GPU |
| Aurora (0.25°) | 3D Swin Transformer + Perceiver | ~32–40GB | впритык на одну GPU, есть fp16-вариант Aurora 1.5 |
| Aurora fine-tuning | — | докам нужен A100 80GB + bf16 | V100 не поддерживает bf16 tensor cores (только Volta fp16/fp32) — нужен fp16 + gradient checkpointing/FSDP на 4 GPU |
| FuXi / FuXi-2.0 | Swin Transformer V2, каскадная | не документировано, вероятно меньше Aurora | вероятно подходит, естественно ансамблить на 4 GPU |
| FengWu | multi-encoder + transformer fusion | не раскрыто | вероятно подходит |
| ClimaX / Stormer | ViT | низкие требования | легко подходит, проще всего дообучать |
| GraphCast | GNN (не transformer) | — | вне scope — не transformer-архитектура |

## Входные поля (Pangu-Weather / Aurora)

- Поверхность: MSLP/msl, U10/u10, V10/v10, T2M/t2m (Aurora 1.5: расширенный набор из 26 переменных)
- Верхние уровни: Z, Q, T, U, V на 13 уровнях давления: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 гПа
- Статика (Aurora): lsm, slt, z (орография)

## Источник данных: ARCO ERA5

```python
import xarray as xr
ds = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    storage_options=dict(token='anon'))
```

**Важно**: имена переменных в ARCO (длинные CDS-имена, напр. `2m_temperature`) не совпадают с именами, которые ждут Pangu/Aurora (короткие GRIB-имена, напр. `t2m`) — нужен маппинг. Готовое решение: `earth2studio.data.ARCO` (NVIDIA), уже делает маппинг и подготовку тензоров под Pangu/Aurora.

Уровни давления: выбрать нужные 13 из 37 доступных в ARCO через `.sel(level=[...])`.

## План по интерпретируемости

Методы, применимые к этим моделям (по убыванию приоритета для старта):

1. **Integrated Gradients** через `captum` (PyTorch) — по данным статьи XAI4Extremes показал лучший результат среди post-hoc методов на transformer-моделях погоды
2. **GeoXplain** (arxiv 2607.05655) — готовая система для визуальных объяснений Aurora: Saliency, Integrated Gradients, RISE, ViT-CX
3. **Ablation по входным переменным/уровням** — занулить/заменить климатологией одну переменную или уровень, сравнить деградацию skill score (RMSE/ACC)
4. **Latent regime analysis** (arxiv 2606.26361, "Does Aurora Encode Atmospheric Structure?") — PCA/UMAP по латентным эмбеддингам Aurora, проверка кластеризации физических режимов
5. **Mechanistic interpretability** (arxiv 2604.20467) — разбор attention-голов и внутренних представлений

## Ссылки

- Pangu-Weather: https://github.com/198808xc/Pangu-Weather
- Aurora: https://github.com/microsoft/aurora
- ARCO ERA5: https://github.com/google-research/arco-era5
- earth2studio (NVIDIA): https://nvidia.github.io/earth2studio
- GeoXplain: https://arxiv.org/pdf/2607.05655
- Does Aurora Encode Atmospheric Structure?: https://arxiv.org/pdf/2606.26361
- Mechanistic Interpretability Tool for AI Weather Models: https://arxiv.org/pdf/2604.20467
- Interpretable ML for Weather and Climate Prediction (Survey): https://arxiv.org/pdf/2403.18864
- XAI4Extremes: https://arxiv.org/html/2503.08163v1

## Ноутбуки

- `01_era5_visualization.ipynb` — визуализация трёх синоптических кейсов (Storm Eunice, волна тепла/блокинг, спокойный день) из ARCO ERA5
- `02_saliency_experiment.ipynb` — occlusion saliency на Pangu-Weather (global per-variable + spatial patch + сравнение с ground truth). Тяжёлые вычисления считаются `run_case_occlusion.py` параллельно на 3 GPU
- `03_output_correlation.ipynb` — корреляционная матрица выходов модели (raw/anomaly, модель vs ground truth, локальная пространственная корреляция)
- `04_aurora_vs_pangu.ipynb` — Aurora (Microsoft, 0.25° pretrained, 1.26B, PyTorch) baseline-прогноз на тех же 3 кейсах, сравнение RMSE и карт с Pangu-Weather и ground truth. Baseline считается `run_case_aurora.py`, параллельно на 3 GPU

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

- `/srv/exw/checkpoints/pangu_weather_24/pangu_weather_24.onnx` — веса модели (общие, не per-project)
- `/srv/exw/data/irina_weather_interpretability/` — входные данные (ERA5-срезы по трём кейсам)
- `/srv/exw/runs/irina_weather_interpretability/` — результаты occlusion-прогонов (`results/*.pkl`)

Симлинки `data` и `results` закоммичены в git (git symlink, mode 120000) — восстанавливаются автоматически при
`git clone`. Только `model_weights/` — сама папка в `.gitignore`, её нужно собрать руками один раз:

```bash
cd weather-interpretability
mkdir -p model_weights
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

## Статус

Три ноутбука реализованы (визуализация, saliency, корреляция выходов). Известные ограничения задокументированы
внутри соответствующих ноутбуков (орографический артефакт Z@850hPa, шум local spatial correlation на малых патчах).
