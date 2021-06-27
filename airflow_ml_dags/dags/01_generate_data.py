import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "cherninkiy",
    "email": ["cherninkiy@mail.ru"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "01_generate_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5)
) as dag:
    generate = DockerOperator(
        image="airflow-generate",
        command="--output-path /opt/airflow/data/raw/{{ ds }} --model-path /data/models/ranfor.pkl",
        network_mode="bridge",
        task_id="generate_data",
        do_xcom_push=False,
        volumes=["/home/dm/MADE-22/ml-in-prod/cherninkiy/airflow_ml_dags/data:/data"]
    )

    generate
