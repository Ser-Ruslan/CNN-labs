"""Сервер: FastAPI (POST /predict, GET /health) и интерфейс Gradio на "/".

Запуск без Docker:  python app/main.py           (из корня проекта, модель берётся из artifacts/torch_best.pt)
В Docker модель лежит в /app/torch_best.pt, путь можно поменять переменной MODEL_PATH.
"""
import io
import os
from pathlib import Path

import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from model import DigitClassifier, image_to_gray, to_mnist

if os.name == "nt":
    # Обход гонки в Gradio под Windows. Холст при каждом нажатии заново загружает те же файлы. Gradio кладёт их в папку
    # с именем по хешу содержимого, а если файл там уже есть, перезаписывает его копированием в фоне. Если в этот момент
    # обработчик читает файл, PIL видит его недописанным ("cannot identify image file"). Файл с тем же хешем и именем
    # уже содержит те же байты, поэтому его можно просто не трогать. На Linux (Colab, Docker) этой проблемы нет.
    import shutil
    import gradio.routes

    def _move_keep_existing(files, destinations):
        for src, dest in zip(files, destinations):
            if os.path.exists(dest):
                try:
                    os.remove(src)
                except OSError:
                    pass
            else:
                shutil.move(src, dest)

    gradio.routes.move_uploaded_files_to_cache = _move_keep_existing

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "torch_best.pt"
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = HERE.parent / "artifacts" / "torch_best.pt"
clf = DigitClassifier(os.environ.get("MODEL_PATH", DEFAULT_MODEL))

app = FastAPI(title="MNIST CNN", description="Распознавание рукописной цифры на картинке")


@app.get("/health")
def health():
    return {"status": "ok", "params": clf.n_params, "cfg": clf.cfg}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Принимает картинку (png/jpg) с одной цифрой, возвращает ответ и вероятности по классам."""
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="не удалось прочитать картинку")
    proba = clf.predict(to_mnist(image_to_gray(img)))
    return {"digit": int(proba.argmax()), "confidence": float(proba.max()),
            "probabilities": {str(d): round(float(p), 5) for d, p in enumerate(proba)}}


def classify(value):
    x28 = to_mnist(image_to_gray(value))
    proba = clf.predict(x28)
    return {str(d): float(proba[d]) for d in range(10)}, np.kron(x28, np.ones((6, 6), np.uint8))   # картинка 28x28, увеличенная в 6 раз


GR5 = int(gr.__version__.split(".")[0]) >= 5
sketch_kw = dict(label="Нарисуйте цифру", type="numpy", image_mode="RGBA", canvas_size=(280, 280))
if GR5:
    sketch_kw["brush"] = gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed")
demo = gr.Interface(
    fn=classify,
    inputs=gr.Sketchpad(**sketch_kw),
    outputs=[gr.Label(num_top_classes=3, label="Ответ сети"), gr.Image(label="Что видит сеть (28x28)", image_mode="L")],
    title="Распознавание рукописных цифр (CNN, MNIST)",
    description=f"PyTorch CNN, {clf.n_params:,} параметров. API: POST /predict, документация: /docs",
    **({"flagging_mode": "never"} if GR5 else {"allow_flagging": "never"}),
)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
