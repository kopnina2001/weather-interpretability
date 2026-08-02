# Weather Model Interpretability

Интерпретируемость transformer-based AI-моделей погоды (Aurora, Pangu-Weather, Stormer).
Кластер «Летний кластер AIRI», 4× V100 32GB.

Основной результат проекта — **причинная карта влияний между входными и выходными полями**,
полученная подменой входных полей на климатологию, и **закон роста ошибки** при плавном
переходе от реальных данных к климатологии.

---

## Структура репозитория

```
├── notebooks/          интерактивный анализ (jupytext percent-format .py)
├── experiments/        прогоны моделей, пишут в results/
│   ├── campaign/         2-летние верификационные кампании (Aurora/Pangu/Stormer)
│   ├── patching/         подмена входных полей на климатологию, 19 переменных
│   ├── dose_response/    линейный блендинг real ↔ климатология
│   └── data/             скачивание ERA5, отбор дат и центров штормов
├── analysis/           results/ → метрики (RMSE, ACC, матрицы)
├── plots/              метрики → картинки
├── figures/            готовые картинки, по темам
├── report/             отчёты и методические заметки (LaTeX + PDF)
├── results →           симлинк на /srv/exw/runs/… (в git не лежит)
├── data →              симлинк на /srv/exw/data/… (в git не лежит)
└── legacy/             отложенные ветки исследования, ничего не удалено
```

`.py` вместо `.ipynb` — чтобы diff читался в git. Открываются как ноутбуки через jupytext
(`jupytext --to ipynb notebooks/climatology.py`), ячейки размечены `# %%`.
Оригинальные `.ipynb` с сохранёнными выводами лежат в `legacy/notebooks/`.

---

## Данные

**ARCO ERA5** — ground truth:
```python
ds = xr.open_zarr('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
                  storage_options=dict(token='anon'))
```
Кэш на 2 года (2024–2025, 6-часовой): `/srv/exw/data/irina_weather_interpretability/era5_2024_2025_6h_{surface,upper}.zarr`

**WeatherBench-2 climatology** (day-of-year × hour-of-day, 1990–2019):
```python
ds = xr.open_zarr('gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr',
                  storage_options=dict(token='anon'))
```
Кэш: `climatology_1990-2019_{surface,upper_500_850_1000}.zarr`.
**Важно**: для upper-air скачаны только уровни 500/850/1000 гПа — полный набор из 13 уровней
не помещался. Все эксперименты с климатологией ограничены этими тремя уровнями.

`chunks=None` в `xr.open_zarr()` — критично, иначе dask даёт OOM на больших срезах.

### Поля
19 переменных: 4 приземных (MSLP, U10, V10, T2M) + 5 upper-air (Z, Q, T, U, V)
на 1000/850/500 гПа.

Z — **сырой геопотенциал Φ в м²/с²**, не геопотенциальная высота (делить на g₀ = 9.80665
для получения гпм). RMSE и ACC инвариантны к этому множителю.

---

## Модели

| Модель | Архитектура | Разрешение | Формат | Особенность |
|---|---|---|---|---|
| Aurora (1.3B) | Perceiver3D encoder + Swin3D U-Net | 0.25°, 13 уровней | PyTorch | 2 входных таймстепа, обрезает сетку до 720 широт |
| Pangu-Weather (256M) | 3D Earth-Specific Transformer | 0.25°, 13 уровней | ONNX | 1 таймстеп, 721 широта, уровни в **убывающем** порядке |
| Stormer | плоский ViT, 24 блока | 1.40625° | PyTorch | нормализованное пространство |

GraphCast — вне scope, не transformer.

### Пределы Aurora на одной V100 32GB (установлено экспериментально)
- `batch=2` → OOM (нужно 6.4 ГБ сверх доступных 31.7). Потолок — batch=1.
- Градиенты → OOM даже с `model.configure_activation_checkpointing()`: forward влезает
  в 19.9 ГБ, но backward запрашивает ещё 15.8 при 11.8 свободных. Нужна карта ≥48 ГБ.
  Поэтому вместо градиентных методов используется подмена входов (см. ниже).

### Три нюанса, из-за которых `aurora.rollout()` не работает напрямую
1. **`Metadata.time` — один элемент, не по одному на входной таймстеп.** Тензоры требуют
   2 времени (t−6ч и t), но `metadata.time` всегда 1-кортеж. Передача 2-элементного
   не бросает ошибку, а тихо задирает память до OOM на многошаговом rollout.
2. **`aurora.rollout()` даёт ~8 ГБ лишнего overhead** против прямого `model.forward()`.
   Обход: ручной rollout через `model.forward()` + `aurora.rollout._advance_batch()`.
3. **Выходная сетка 720 широт, не 721** — модель срезает южный полюс. Нужен
   `batch.crop(model.patch_size)` один раз перед циклом, иначе `_advance_batch` падает.

С `torch.autocast('cuda', dtype=torch.float16)` (не `model.half()` — ломает проверку
на `lat`/`lon`, они должны остаться fp32) — стабильно 19.6 ГБ на шаг.

---

## Методы и результаты

### 1. Верификационная кампания
`experiments/campaign/` → `notebooks/model_comparison_*.py`, `notebooks/climatology.py`

96 инициализаций (2024–2025, 4/месяц) × 3 модели × 8 лидов (6–48ч).
Метрики: RMSE, ACC против WB2-климатологии, bias, variance ratio, спектры.
Карты ошибок для синоптических кейсов (тайфун Bebinca, Китай), включая сравнение
с климатологическим baseline.

Картинки: `figures/model_comparison/`, `figures/climatology/`, `figures/data_overview/`

### 2. Матрица влияний: подмена входа на климатологию
`experiments/patching/` → `analysis/compute_19var_matrix.py` → `figures/patching_matrices/`

Причинная абляция: подменить **одно** входное поле на климатологию, измерить, как
испортились все 19 выходных. 48 дат 2024 года (дни 4/11/18/25, часы 00/06/12/18),
леды +6ч и +24ч. Даёт матрицу 19×19 — прокси для якобиана без градиентов.

**Находки:**
- Матрица сильно **асимметрична**: Z → ветер в 8–20 раз сильнее обратного,
  Z500 → Q500 до 126 раз.
- **Z500 слабо влияет сам на себя** — модель восстанавливает его из других полей.
  **MSLP, наоборот, незаменим** — несёт уникальную информацию.
- Aurora и Pangu структурно согласуются (r = 0.913), но у Pangu приземные self-effects
  в 2.6–4 раза выше, а урон по картам вдвое больше и гораздо менее локализован.

### 3. Dose-response: закон роста ошибки
`experiments/dose_response/` → `analysis/analyse_dose_z1000.py` → `plots/plot_dose_panels*.py`

Вместо бинарной подмены — плавный переход:

    x_α = (1 − α)·x_real + α·c_clim ,   α ∈ {0, 0.2, 0.4, 0.6, 0.8, 1}

Подменяется Z1000, те же 48 дат. Отвечает на вопрос, законен ли бинарный эксперимент
как прокси градиента.

**Находки:**
- Кривая **слегка выпуклая**, насыщения нет → бинарная матрица оправдана как прокси.
  Нормированная RMSE ×2.37 при +6ч и ×1.39 при +24ч.
- Сильнее всего страдает **термодинамика, не ветер**: T1000 (k = 0.69), Q1000, T2M.
  При α = 1 ошибка T1000 достигает 0.81 СКО аномалии — прогноз почти теряет скилл.
- Физически: Z1000 ≈ приземное давление, связано с T гипсометрически напрямую;
  ветер восстанавливается из геострофического баланса по другим уровням.

### 4. Bias maps
`plots/plot_bias_map.py` (одна дата), `plots/composite_bias_maps.py` (композит) →
`figures/bias_maps/`

Попиксельно: |ошибка| baseline vs подменённого прогона, разность локализует урон.
Композит — временная RMSE по 48 датам в каждом узле сетки отдельно (поэтому cos φ
не входит: по глобусу ничего не агрегируется).

При α = 1, +6ч: T1000 хуже на 100% узлов (RMSE ×6.02), Q1000 ×2.92, U/V1000 ×2.7.
Урон глобальный, максимум в шторм-треках средних широт обоих полушарий.

---

## Метрики

Все пространственные метрики взвешены по площади ячейки, `w = cos φ / Σ cos φ`
(конвенция WeatherBench-2 / ECMWF):

```
RMSE_w = sqrt( Σ_s w_s (a_s − b_s)² )
σ_w(x) = sqrt( Σ_s w_s (x_s − Σ w x)² )
ACC_w  = Σ w·a'·b' / sqrt( Σ w·a'² · Σ w·b'² ),   a' = pred − clim,  b' = truth − clim
```

**Нормировка RMSE — важная тонкость.** Делить можно на два разных СКО:

| знаменатель | смысл | опорная точка |
|---|---|---|
| σ(правда) | пространственный разброс всего поля | нет |
| σ(аномалии) = σ(правда − клим) | разброс того, что реально надо предсказать | **y = 1 — прогноз не лучше климатологии** |

Отношение σ(правда)/σ(аномалии) меняется от 1.0 (V500 — климатологическое среднее
меридионального ветра ≈ 0) до 5.8 (T2M — доминирует градиент экватор–полюс). Поэтому
нормировка на σ(правда) систематически занижает термодинамические поля относительно
ветра, и топ-5 самых чувствительных полей меняется полностью
(корреляция рангов 0.75 при +6ч, 0.55 при +24ч).

**Корректнее σ(аномалии)** — у неё есть опорная точка. См. `analysis/compare_sigma.py`,
обе версии картинок сохранены для сравнения.

Полный вывод формул — `report/metrics_methodology.pdf`.

---

## legacy/

Отложенные ветки. Ничего не удалено — код рабочий, результаты воспроизводимы.

| папка | что |
|---|---|
| `notebooks/` | оригинальные `.ipynb` со всеми выводами + конвертации отложенных |
| `skeleton/` | interpolative (skeleton) decomposition hidden state и attention |
| `tensor/` | Tucker, NMF, CKA, RepE-style PCA |
| `saliency/` | occlusion saliency на Pangu |
| `tests/` | отладочные прогоны |
| `v1/` | первые версии кампаний, вытесненные `*_v2` (в v2 добавлены bias, variance ratio, сохранение полей) |

**Честные negative results** этой ветки задокументированы наравне с позитивными:
- skeleton на **hidden state**: landmarks садятся на центр тайфуна (850 гПа vorticity
  extremum) — **выжило** после quiet-baseline контроля, подтверждено на второй
  архитектуре (Stormer), знаковый тест p = 0.031 по 6 штормам с парными контролями.
- skeleton на **attention**: landmarks садятся на края домена — артефакт выбора столбцов
  в QR, а не физика. Отброшено.
- **RepE-PCA** на разнице storm−quiet: главная компонента оказалась сезонным конфаундом.
- **spectrum / effective rank**: storm и quiet почти неразличимы.
- **NMF** на attention: структура чище, чем у skeleton, но свойство архитектурное,
  а не storm-specific.

---

## Окружение

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

**GPU + ONNX**: `onnxruntime-gpu` не находит CUDA-библиотеки сам. Добавить в конец
`~/venv/bin/activate` (именно туда — переменная, выставленная после старта Python,
не подхватывается):

```bash
export LD_LIBRARY_PATH=$(find "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia" \
  -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')$LD_LIBRARY_PATH
```

Отдельный system CUDA toolkit не нужен — библиотеки ставит `torch`.

---

## Общая файловая система `/srv/exw`

Тяжёлые артефакты не в git, конвенция команды — `<username>_<project>`:

- `/srv/exw/checkpoints/pangu_weather_{6,24}/…onnx` — веса (общие, без префикса)
- `/srv/exw/data/irina_weather_interpretability/` — ERA5 + климатология
- `/srv/exw/runs/irina_weather_interpretability/` — результаты прогонов

Симлинки `data` и `results` закоммичены (git mode 120000), восстанавливаются при clone.
`model_weights/` в `.gitignore`, собрать руками один раз:

```bash
mkdir -p model_weights
ln -s /srv/exw/checkpoints/pangu_weather_6/pangu_weather_6.onnx model_weights/
ln -s /srv/exw/checkpoints/pangu_weather_24/pangu_weather_24.onnx model_weights/
```

**Дисциплина по месту**: диск переполнялся один раз, положив 4 GPU-воркера и повредив
4 npz. Скрипты пишут атомарно (`.tmp` + `os.replace`), проверяют точное число массивов
в файле при чтении и умеют продолжать с места обрыва. `results/patching/` — 200 ГБ,
следить за `df -h`.

---

## Workflow

**В `main` пишет только владелец репозитория.** Остальные — через ветку и PR:

```bash
git checkout -b <имя>/<фича>
git push -u origin <имя>/<фича>
```

---

## Ссылки

- Pangu-Weather: https://github.com/198808xc/Pangu-Weather
- Aurora: https://github.com/microsoft/aurora
- Stormer: https://github.com/tung-nd/stormer
- ARCO ERA5: https://github.com/google-research/arco-era5
- WeatherBench2: https://github.com/google-research/weatherbench2
- Advection Heads in an Atmosphere Foundation Model — референс для `legacy/skeleton`
- Representation Engineering (RepE): https://arxiv.org/abs/2310.01405
- CKA, Kornblith et al. 2019: https://arxiv.org/abs/1905.00414
- DEIM (Chaturantabut & Sørensen 2010) — предок skeleton-подхода, sensor placement
- Interpretable ML for Weather and Climate Prediction (survey): https://arxiv.org/pdf/2403.18864
