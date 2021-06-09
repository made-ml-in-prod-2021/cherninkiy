from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

OUTPUT_DIR = Variable.get("OUTPUT_DIR")

default_args = {
    "owner": "cherinkiy",
    "email": "cherinkiy@mail.ru",
    "email_on_failure": False,
    "email_on_retry": False,
    "retry_delay": timedelta(minutes=5),
    "retries": 1,
}

with DAG(
    "01_generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
) as dag:
    generate_data = DockerOperator(
        image="airflow-pipeline",
        command="--output_dir '/data/raw/{{ ds }}'",
        network_mode="bridge",
        task_id="generate-data",
        do_xcom_push=False,
        volumes=[f"/data/raw/{{ ds }}:/data"],
    )
