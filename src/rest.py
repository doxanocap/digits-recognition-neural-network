import logging
import base64
import uvicorn
import network

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

app = FastAPI()
neural_network = network.Network

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class ImageRequest(BaseModel):
    base64: str


class ErrorResponse(BaseModel):
    message: str


@app.post("/image")
async def process_image(image_request: ImageRequest):
    global neural_network
    try:
        image_bytes = base64.b64decode(image_request.base64)

        image = Image.open(BytesIO(image_bytes))
        image = image.convert('L')
        image.save("../data/image.png")

        resized_image = image.resize((28, 28))
        image_data = np.asarray(resized_image)
        flattened_image = image_data.reshape(-1, 1)

        results = neural_network.feed_forward(flattened_image)
        return {"number": int(np.argmax(results))}

    except Exception as e:
        print(str(e))
        response = ErrorResponse(message="internal server error")
        return JSONResponse(content=jsonable_encoder(response), status_code=500)


@app.options("/image")
async def options_image():
    return {"message": "OPTIONS request received. Please use POST method for image upload."}


def initREST(initialized_nn):
    global neural_network
    neural_network = initialized_nn
    uvicorn.run("rest:app", host="0.0.0.0", port=8000, reload=False)
