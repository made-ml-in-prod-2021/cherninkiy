from airflow.models import DagBag
import pytest
import os

os.chdir(os.path.abspath(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def dag_bag():
    dag_bag = DagBag(dag_folder='../dags/', include_examples=False)
    return dag_bag
