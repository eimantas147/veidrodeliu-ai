# Eksperimentu atkartojimas be kameros: docker build -t veidrodeliai . && docker run --rm -v $PWD/results:/app/results veidrodeliai
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir numpy pandas scikit-learn matplotlib joblib
COPY src/ src/
CMD python src/generate_data.py && python src/train_eval.py && python src/make_figures.py
