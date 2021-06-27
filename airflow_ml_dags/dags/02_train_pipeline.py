from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor
from airflow.models.baseoperator import chain
from airflow.sensors.python import PythonSensor
import os

default_args = {
    "owner": "cherninkiy",
    "email": ["cherninkiy@mail.ru"],
    "retries": 10,
    "retry_delay": timedelta(minutes=5),
}


def _wait_for_file(path):
    print(path)
    print(os.path.exists(path))
    print(os.path.abspath(os.curdir))
    return os.path.exists(path)


with DAG(
        "02_train_pipeline",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5)
) as dag:

    wait_data = PythonSensor(
        task_id="wait_data",
        python_callable=_wait_for_file,
        op_kwargs={'path':'/opt/airflow/data/raw/{{ ds }}/data.csv'},
        timeout=10,
        poke_interval=10,
        retries=2,
        mode="poke",
    )

    wait_target = PythonSensor(
        task_id="wait_target",
        python_callable=_wait_for_file,
        op_kwargs={'path':'/opt/airflow/data/raw/{{ ds }}/target.csv'},
        timeout=10,
        poke_interval=10,
        retries=2,
        mode="poke",
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        task_id="preprocess_data",
        do_xcom_push=False,
        volumes=["/home/dm/MADE-22/ml-in-prod/cherninkiy/airflow_ml_dags/data:/data"]
    )

    split = DockerOperator(
        image="airflow-split",
        command="--data-path /data/processed/{{ ds }}",
        task_id="split_data",
        do_xcom_push=False,
        volumes=["/home/dm/MADE-22/ml-in-prod/cherninkiy/airflow_ml_dags/data:/data"]
    )

    train = DockerOperator(
        image="airflow-train",
        command="--data-path /data/processed/{{ ds }} --model-name RandomForestClassifier --model-path /data/models/{{ ds }}/flow_model.pkl",
        task_id="train_pipeline",
        do_xcom_push=False,
        volumes=["/home/dm/MADE-22/ml-in-prod/cherninkiy/airflow_ml_dags/data:/data"]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="--data-path /data/processed/{{ ds }} --model-path /data/models/{{ ds }}/flow_model.pkl",
        task_id="validate_pipeline",
        do_xcom_push=False,
        volumes=["/home/dm/MADE-22/ml-in-prod/cherninkiy/airflow_ml_dags/data:/data"]
    )

    wait_data >> wait_target >> preprocess >> split >> train >> validate
