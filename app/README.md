# Приложение для распознавания цифр

Сервер на FastAPI с интерфейсом Gradio. Модель - лучшая по валидации CNN на PyTorch из `lab2_cnn_mnist.ipynb`
(тетрадка сохраняет её в `artifacts/torch_best.pt`).

* `/` - интерфейс: рисуете цифру мышкой, сеть показывает три самых вероятных ответа и картинку 28x28, которую она получила;
* `POST /predict` - API, принимает картинку с одной цифрой и возвращает JSON `{"digit", "confidence", "probabilities"}`;
* `GET /health` - проверка, что сервер жив, и параметры модели;
* `/docs` - автоматическая документация API (Swagger).

## Без Docker

```bash
pip install torch numpy scipy pillow fastapi uvicorn python-multipart gradio
python app/main.py
```

Открыть http://127.0.0.1:8000/

## Docker

Команды выполняются из корня проекта: в образ копируются `app/` и `artifacts/torch_best.pt`.

```bash
docker build -f app/Dockerfile -t mnist-cnn .
docker run -p 8000:8000 mnist-cnn
```

Проверка API:

```bash
curl -F "file=@digit.png" http://localhost:8000/predict
```

Картинка может быть любого размера, тёмная цифра на светлом фоне или наоборот. Перед подачей в сеть она приводится к виду MNIST:
обрезка по краям цифры, большая сторона 20 px, центр масс в центре кадра 28x28.
