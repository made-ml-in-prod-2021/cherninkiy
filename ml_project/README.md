ml_project
==============================

ML in production course practice

Установка:

    pip install -r requirements.txt
    python setup.py install

Обучение модели:

    python run_pipeline.py train conf/pipelene.yml

Предсказание по обученной модели:

    python run_pipeline.py predict configs/predict_config.yml

Запуск тестов:

    pytest tests/

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
