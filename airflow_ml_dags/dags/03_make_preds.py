from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.python import PythonSensor
import os

default_args = {
    "owner": "cherninkiy",
    "email": ["cherninkiy@mail.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _wait_for_file(path):
    print(path)
    print(os.path.exists(path))
    print(os.path.abspath(os.curdir))
    return os.path.exists(path)


with DAG(
        "03_make_preds",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5)
) as dag:
    wait_data = PythonSensor(
        task_id="wait_for_data",
        python_callable=_wait_for_file,
        op_kwargs={'path':'/opt/airflow/data/processed/{{ ds }}/data.csv'},
        timeout=10,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--data-path /data/processed/{{ ds }}/data.csv --model-path /data/models/{{ ds }}/flow_model.pkl --output-path /data/predictions/{{ ds }}",
        task_id="make_preds",
        do_xcom_push=False,
        volumes=["/home/dm/MADE-22/ml-in-prod/cherninkiy/airflow_ml_dags/data:/data"]
    )

    wait_data >> predict
