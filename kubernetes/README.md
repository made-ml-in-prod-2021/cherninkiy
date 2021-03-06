kubernetes_practice
==============================

Что сделано:
- ✓ 0) Установите kubectl
- ✓ 1) Разверните kubernetes (5 баллов)

Кластер поднял локально (https://kind.sigs.k8s.io/docs/user/quick-start/)

<img src="https://github.com/made-ml-in-prod-2021/cherninkiy/blob/homework4/kubernetes/screenshots/cluster_info.png?raw=true" alt="cluster_info" width="320"/>

- ✓ 2) Напишите простой pod manifests для вашего приложения, назовите его online-inference-pod.yaml (4 балла)

<img src="https://github.com/made-ml-in-prod-2021/cherninkiy/blob/homework4/kubernetes/screenshots/online_inference.png?raw=true" alt="online_inference" width="320"/>

- ✓ 2а) Пропишите requests/limits и напишите зачем это нужно в описание PR (2 балла)
  
Requests - те требования к железу, которые необходимы для запуска образа для стабильной его работы. 

Limits - максимальные требования, для эффективной загрузки кластера при запуске на нем нескольких образов.

- ✓ 3) Модифицируйте свое приложение так, чтобы оно стартовало не сразу(с задержкой секунд 20-30) и падало спустя минуты работы. 
Добавьте liveness и readiness пробы, посмотрите что будет происходить. (3 балла)
   
Readiness - образ прогружен, все необходимые сервисы работают, нужно для учета времени, 
  необходимого на инициализацию сервисов внутри контейнера.

Liveness - проверка работоспособности сервиса, необходимо для определения целесообразности перезапуска контейнера.
  
- ✓ 4) Создайте replicaset, сделайте 3 реплики вашего приложения. (3 балла)

Если сменить образ и увеличить число реплик, старые реплики продолжают работать
со старым образом, новые запускаются на новом. Если уменьшить количество реплик, то последние реплики гасятся

- x 5) Опишите деплоймент для вашего приложения.  (3 балла)

Самооценка: 17 баллов.



