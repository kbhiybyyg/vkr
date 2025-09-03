# Проект: Веб‑приложение для прогнозирования свойств композитов

## Описание
Веб‑приложение на **Flask** с фронтендом и CLI‑утилитами.  
Назначение: прогнозирование свойств композитных материалов и рекомендация оптимального соотношения матрица‑наполнитель на основе обученных моделей.

## Структура проекта
```
vkr/
├── app/                     # Flask-приложение (backend)
│   ├── __init__.py
│   ├── routes.py
│   ├── validators.py
│   └── utils.py
│
├── cli/                     # консольные утилиты
│   ├── predict_properties.py
│   └── recommend_ratio.py
│
├── models/                  # обученные модели
│   ├── properties/          # sklearn-модели для предсказаний
│   └── ratio/               # TF-модель для рекомендации
│
├── frontend/                # фронтенд HTML/JS/CSS
│   ├── public/
│   └── src/
│
├── data/                    # примеры входных данных
├── notebooks/               # Jupyter/py-скрипты для обучения
│   └── ВКР.ipynb
│
├── requirements.txt
├── run.py                   # запуск Flask
└── README.md
```

## Установка
1. Клонировать репозиторий.
   ```bash
   git clone <name>
   ```
2. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Использование

### CLI
**Прогноз свойств:**
```bash
python cli/predict_properties.py --input data/samples.csv --output out.csv
```

**Рекомендация соотношения:**
```bash
python cli/recommend_ratio.py --input data/samples.csv --output ratio.csv
```

### Flask
```bash
python run.py
```
Приложение поднимется на `http://localhost:8000`.

Доступные маршруты:
- `/api/health` — проверка состояния.
- `/api/predict` — POST‑запрос для прогноза свойств.

### Фронтенд
-  Файлы лежат в `frontend/public`, Flask настроен на их раздачу.

## Требования
- Python 3.10+
- Flask
- pandas, numpy
- scikit-learn, joblib
- tensorflow

## Авторы
- Выпускная квалификационная работа  
- Автор: Богуславский Дмитрий Владимирович
- Почта: 99-rr99@mail.ru
