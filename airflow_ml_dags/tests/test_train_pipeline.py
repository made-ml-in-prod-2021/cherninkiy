def test_dag_loaded(dag_bag):
    dag = dag_bag.dags.get("02_train_pipeline")

    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 6
    assert dag.tasks[0].task_id == 'wait_data'
    assert dag.tasks[1].task_id == 'wait_target'
    assert dag.tasks[2].task_id == 'preprocess_data'
    assert dag.tasks[3].task_id == 'split_data'
    assert dag.tasks[4].task_id == 'train_pipeline'
    assert dag.tasks[5].task_id == 'validate_pipeline'
