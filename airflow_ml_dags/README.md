airflow_ml_dags
==============================

Workflow модели предсказания болезней сердца по базе признаков Cleveland Clinic.

Архитектурные решения:

- реализация на платформе Apache Airflow

- виртуализация с прменением контейнеров Docker

- использование модели предсказания данных ml-pipeline (Homework1)

Рабочий процесс:

- генерация синтетических данных
- предобработка, выделение тестового и проверочного наборов, обучение модели, валидация
- предсказание данных на обученной модели

<img src="https://github.com/made-ml-in-prod-2021/cherninkiy/blob/homework3/airflow_ml_dags/screenshots/dags.png?raw=true" alt="dags" width="320"/>

Запуск workflow:

    docker compose up --build

Что сделано:

- ✓ 1) Реализуйте dag, который генерирует данные для обучения модели (5 баллов)

<img src="https://github.com/made-ml-in-prod-2021/cherninkiy/blob/homework3/airflow_ml_dags/screenshots/01_generate_data.png?raw=true" alt="01_generate_data" width="320"/>

- ✓ 2) Реализуйте dag, который обучает модель еженедельно, используя данные за текущий день (10 баллов)

<img src="https://github.com/made-ml-in-prod-2021/cherninkiy/blob/homework3/airflow_ml_dags/screenshots/02_train_pipeline.png?raw=true" alt="02_train_pipeline" width="320"/>

- ✓ 3) Реализуйте dag, который использует модель ежедневно (5 баллов)

<img src="https://github.com/made-ml-in-prod-2021/cherninkiy/blob/homework3/airflow_ml_dags/screenshots/03_make_preds.png?raw=true" alt="03_make_preds" width="320"/>

- ✓ 3а) Реализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения (3 доп балла)
- ✓ 4) Все даги реализованы только с помощью DockerOperator (10 баллов) 
- ✓ 5) Протестируйте ваши даги (5 баллов) 
- x 6) В docker compose так же настройте поднятие mlflow и запишите туда параметры обучения, метрики и артефакт(модель) (5 доп баллов)
- x 7) вместо пути в airflow variables  используйте апи Mlflow Model Registry (5 доп баллов)
- x 8) Настройте alert в случае падения дага (3 доп. балла)
- ✓ 9) традиционно, самооценка (1 балл)

Самооценка: 24 балла, hard deadline


Структура проекта

    ├── LICENSE
    ├── README.md                  <- The top-level README for developers using this project
    ├── docker-compose.yml         <- Makefile with commands like `make data` or `make train`
    ├── dags
    │   ├── 01_generate_data.py    <- Generating data DAG script
    │   ├── 02_train_pipeline.py   <- Trainning model DAG script
    │   └── 03_make_preds.py       <- Making predictions DAG script
    ├── images                     <- Workflow docker images
    │   ├── airflow-docker
    │   │   └── Dockerfile
    │   ├── airflow-ml-base
    │   │   ├── requirements.txt
    │   │   └── Dockerfile
    │   ├── airflow-generate
    │   │   ├── generate.py
    │   │   └── Dockerfile
    │   ├── airflow-preprocess
    │   │   ├── preprocess.py
    │   │   └── Dockerfile
    │   ├── airflow-split
    │   │   ├── split.py
    │   │   └── Dockerfile
    │   ├── airflow-train
    │   │   ├── train.py
    │   │   └── Dockerfile
    │   ├── airflow-validate
    │   │   ├── validate.py
    │   │   └── Dockerfile
    │   ├── airflow-predict
    │   │   ├── predict.py
    │   │   └── Dockerfile
    │   ├── tests                  <- Pytest scripts
    │   │   ├── conftest.py
    │   │   ├── test_generate_data.py
    │   │   ├── test_train_pipeline.py
    │   │   └── test_make_preds.yaml
    └── screenshorts               <- Workflow screenshorts
        ├── dags.png
        ├── 01_generate_data.pkl
        ├── 02_train_pipeline.pkl
        └── 03_make_preds.pkl
