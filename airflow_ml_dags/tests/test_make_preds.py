def test_dag_loaded(dag_bag):
    dag = dag_bag.dags.get('03_make_preds')

    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 2
