FROM python:3.9.5-slim

RUN apt update && apt install -y git

ENV VIRTUAL_ENV "/venv"
ENV PATH "$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV

RUN mkdir -p /ml_pipeline
COPY ./* /ml_pipeline/

WORKDIR /ml_pipeline
RUN pip install -r ./requirements.txt

ENTRYPOINT ["python", "./generate_data.py"]