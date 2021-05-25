ml_project
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

    docker pull cherninkiy/made-ml-in-prod-2021-v2:latest
    docker run -p 8080:8080 cherninkiy/made-ml-in-prod-2021-v2:latest

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
- x 8) напишите в readme корректные команды docker pull/run (1 балл)
- x 9) Проведите самооценку, опишите, в какое кол-во баллов по вашему мнению стоит оценить вашу работу и почему (1 балл доп баллы) 

Самооценка: 20 баллов, все по ТЗ


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
