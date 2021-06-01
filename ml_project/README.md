ml_project
==============================

Модуль для решения задачи предсказания болезней сердца по базе признаков Cleveland Clinic.

Описание признаков: https://www.kaggle.com/ronitf/heart-disease-uci

Архитектурные решения:
- проект имеет модульную структуру
- для предсказания используются модели на основе логистической регрессии или случайного леса
- конфигурирование модуля осуществляется с помощью конфига в yaml
- для извлечения признаков используется кастомный трансформер 

Установка:

    pip install -r requirements.txt
    python setup.py install

Обучение модели:

    python run_pipeline.py train conf/pipeline.yml

Предсказание по обученной модели:

    python run_pipeline.py predict conf/predict_config.yml

Запуск тестов:

    pytest tests/

Что сделано:
- ✓ -2) Назовите ветку homework1 (1 балл)
- ✓ -1) Положите код в папку ml_project
- ✓ 0) В описании к пулл-реквесту описаны основные "архитектурные" и тактические решения. (2 балла)
- ✓ 1) Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками (2 баллов)
       Можете использовать не ноутбук, а скрипт, который сгенерит отчет (за это + 1 балл)
- ✓ 2) Проект имеет модульную структуру(не все в одном файле =)) (2 баллов)
- ✓ 3) Использованы логгеры (2 балла)
- ✓ 4) Написаны тесты на отдельные модули и на прогон всего пайплайна (3 баллов)
- x 5) Для тестов генерируются синтетические данные, приближенные к реальным (3 баллов)
- ✓ 6) Обучение модели конфигурируется с помощью конфигов в json или yaml (3 балла)
- ✓ 7) Используются датаклассы для сущностей из конфига, а не голые dict (3 балла) 
- ✓ 8) Используйте кастомный трансформер(написанный своими руками) и протестируйте его(3 балла)
- ✓ 9) Обучите модель, запишите в readme как это предлагается (3 балла)
- ✓ 10) напишите функцию predict, которая примет на вход артефакт/ы от обучения, 
  тестовую выборку(без меток) и запишет предикт (3 балла)  
- ✓ 11) Используется hydra (https://hydra.cc/docs/intro/) (3 балла — доп баллы)
- x 12) Настроен CI(прогон тестов, линтера) на основе github actions (3 балла — доп баллы)
- ✓ 13) Проведите самооценку, опишите, в какое кол-во баллов по вашему мнению стоит оценить вашу работу и почему (1 балл доп баллы) 

Самооценка: 31 балл, все по ТЗ


Структура проекта

	├── LICENSE
	├── Makefile                  <- Makefile with commands like `make data` or `make train`
	├── README.md                 <- The top-level README for developers using this project
	│
	├── conf
	│   ├── data
	│   │   └── data.yaml         <- Data configuration
	│   │
	│   ├── features
	│   │   └── features.yaml     <- Features configuration
	│   │
	│   ├── model
	│   │   └── model.yaml        <- Model configuration
	│   │
	│   ├── transformer
	│   │   └── transformer.yaml  <- Transformer configuration
	│   │
	│   └── pipeline.yaml         <- Pipline configuration
	│
	├── data
	│   └── raw                   <- The original, immutable data dump
	│
	├── docs                      <- A default Sphinx project; see sphinx-doc.org for details
	│
	├── models
	│   ├── logreg.pkl            <- Logistic regression model
	│   └── ranfor.pkl            <- Random forest model
	│
	├── notebooks
	│   └── EDA.ipynb             <- Exploratory data analisys (EDA)
	│
	├── reports
	│   └── EDA-report.ipynb      <- EDA-report.html
	│
	├── requirements.txt          <- The requirements file for reproducing the analysis environment
	│
	├── setup.py                  <- Make this project pip installable with `pip install -e`
	│
	├── tox.ini                   <- tox file with settings for running tox; see tox.readthedocs.io
	│
	└── ml_pipeline               <- package directory
        ├── src                   <- Source code for use in this project
        │   ├── __init__.py       <- Makes src a Python module
        │   ├── data              <- Scripts to download or generate data
        │   │   └── utils.py
        │   │
        │   ├── entities          <- Entities classes.
        │   │   ├── data_params.py
        │   │   ├── feature_params.py
        │   │   ├── model_params.py
        │   │   ├── transformer_params.py
        │   │   └── pipline_params.py
        │   │
        │   ├── features          <- Scripts to turn raw data into features for modeling
        │   │   └── build_features.py
        │   │
        │   ├── models            <- Scripts to train models and then use trained models to make predictions
        │   │   ├── predict_model.py
        │   │   └── train_model.py
        │   │
        │   ├── train_model.py    <- Scripts to make train pipline
        │   └── predict_model.py  <- Scripts to make prediction pipline
        │
        ├── tests                 <- Pytest scripts
        │   ├── test_data_utils.py
        │   ├── test_build_features.py
        │   ├── test_train_model.py
        │   ├── test_predict_model.py
        │   ├── test_train_pipline.py
        │   └── test_predict_pipline.py
    	│
        └── run_pipeline.py       <- Script to run pipeline


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
