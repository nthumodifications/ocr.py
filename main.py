from time import time
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def hello():
    return {'res': 'pong', "time": time()}