from fastapi import FastAPI
from config.settings import get_settings, setup_logging

app = FastAPI(title="agri-harvest API")
setup_logging(get_settings())

@app.get('/health')
def health():
    return {"status": "ok"}
