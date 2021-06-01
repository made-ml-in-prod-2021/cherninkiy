FROM python:3.9.5-slim

ENV VIRTUAL_ENV "/venv"
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

RUN apt update
RUN apt install -y git make
RUN python -m venv $VIRTUAL_ENV

RUN mkdir -p /srv/ml_pipeline
COPY ./ /srv/ml_pipeline/
WORKDIR /srv/ml_pipeline

RUN make requirements
RUN python setup.py install

CMD ["uvicorn", "ml_app.app:app", "--host", "0.0.0.0", "--port", "8080"]