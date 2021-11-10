from fastapi import FastAPI

from faas.storage import list_models, read_model

app = FastAPI()


@app.get("/")
def read_root():
    return list_models()


@app.get("/models/{model_key}")
def read_item(model_key: str):
    return read_model(model_key)
