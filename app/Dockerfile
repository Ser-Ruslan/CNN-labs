# Сборка из корня проекта (нужен artifacts/torch_best.pt, его создаёт тетрадка):
#   docker build -f app/Dockerfile -t mnist-cnn .
#   docker run -p 8000:8000 mnist-cnn
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000

# PyTorch только для CPU: образ получается в несколько раз меньше, чем с CUDA
COPY app/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY app/model.py app/main.py ./
COPY artifacts/torch_best.pt ./torch_best.pt

EXPOSE 8000
CMD ["python", "main.py"]
