FROM python:3.9.5-slim

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY src /src
COPY conf /conf
COPY models /models
COPY app.py /app.py

WORKDIR .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]