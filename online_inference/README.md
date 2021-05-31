online_inference
==============================

REST-сервис предсказания болезней сердца по базе признаков Cleveland Clinic.

Описание признаков: https://www.kaggle.com/ronitf/heart-disease-uci

Архитектурные решения:
- используется фреймворк FastAPI
- реализована валидация значений входных данных
- порядок полей во входных данных может быть произвольный
- для оптимизации размера docker image использован образ python:3.9.5-slim (linux/amd64) - 41.47 MB

Репозиторий https://hub.docker.com/u/cherninkiy

Установка образа:

    docker pull cherninkiy/made-ml-in-prod-2021-v3:latest
    docker run -p 8080:8080 cherninkiy/made-ml-in-prod-2021-v3:latest

API предсказателя:

    http://HOST:8080         - точка входа
    http://HOST:8080/status  - состояние пайплайна
    http://HOST:8080/predict - GET-запрос на предикт

Что сделано:
- ✓ 0) Ветку назовите homework2, положите код в папку online_inference
- ✓ 1) Оберните inference вашей модели в rest сервис, , должен быть endpoint /predict (3 балла)
- ✓ 2) Напишите тест для /predict (pytest) (3 балла)
- ✓ 3) Напишите скрипт, который будет делать запросы к вашему сервису (2 баллов)
- ✓ 4) Сделайте валидацию входных данных (3 балла — доп баллы)
- ✓ 5) Напишите dockerfile, соберите на его основе образ и запустите локально контейнер (4 балл)
- ✓ 6) Оптимизируйте размер docker image (3 доп балла)
- ✓ 7) Опубликуйте образ в https://hub.docker.com/  (2 балла)
- ✓ 8) Напишите в readme корректные команды docker pull/run (1 балл)
- ✓ 9) Проведите самооценку, опишите, в какое кол-во баллов по вашему мнению стоит оценить вашу работу и почему (1 балл доп баллы) 

Самооценка: 20 баллов, все по ТЗ


Структура проекта

    ├── LICENSE
    ├── Makefile                 <- Makefile with commands like `make data` or `make train`
    ├── README.md                <- The top-level README for developers using this project
    ├── tox.ini                  <- tox file with settings for running tox; see tox.readthedocs.io
    ├── docs                       <- A default Sphinx project; see sphinx-doc.org for details
    ├── data
    │   └── test.csv               <- Testing data
    ├── conf
    │   ├── data
    │   │   └── data.yaml        <- Data configuration
    │   ├── features
    │   │   └── features.yaml    <- Features configuration
    │   ├── model
    │   │   └── model.yaml       <- Model configuration
    │   ├── transformer
    │   │   └── transformer.yaml   <- Model configuration
    │   ├── ranfor_pipeline.yaml   <- RandomForest pipline
    │   └── logreg_pipeline.yaml   <- LogisticRegression pipline
    ├── models
    │   ├── logreg.pkl             <- Logistic regression model
    │   ├── ranfor.pkl             <- Random forest model
    │   └── transformer.pkl        <- Feature builder
    ├── requirements.txt           <- The requirements file for reproducing the analysis environment
    ├── setup.py                   <- Make this project pip installable with `pip install -e`
    ├── Dockerfile                 <- Docker configuration
    └── ml_app                     <- Package scripts
        ├── __init__.py            <- Makes a Python module
        ├── src                    <- Source code for use in this project
        │   ├── __init__.py        <- Makes src a Python module
        │   ├── entities           <- Entities classes.
        │   │   └── app_params.py
        │   ├── app.py             <- Script to run service
        │   └── app_request.py     <- Script to make request
        ├── tests                  <- Pytest scripts
        │   └── test_app.py
        ├── app.py                 <- REST-service script
        └── app_request.py         <- Request script
