import base64
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

app = FastAPI()
network = sgd.Network
image_title = "curr"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    base64: str


class ErrorResponse(BaseModel):
    message: str


@app.post("/image")
async def process_image(image_request: ImageRequest):
    global network, image_title
    try:
        image_bytes = base64.b64decode(image_request.base64)

        image = Image.open(BytesIO(image_bytes))
        image = image.convert('L')

        image_title = "curr" if image_title == "curr" else "prev"
        resized_image = image.resize((28, 28))
        resized_image.save("../data/tests/image-" + image_title + ".png", "PNG")

        image_data = np.asarray(resized_image)
        flattened_image = image_data.reshape(-1, 1)

        output = network.eval(flattened_image)
        percentages = [v[0] * 100 / float(np.sum(output)) for v in output]

        plt.bar(range(10), percentages, color="blue")
        plt.xticks(range(1, 11))
        plt.xlabel("number")
        plt.ylabel("similarity")

        plt.savefig(f"../data/tests/plot{image_title}.png", format="png")
        plt.clf()

        return {
            "number": int(np.argmax(output)),
            "image_title": image_title
        }

    except Exception as e:
        print("err", str(e))
        response = ErrorResponse(message="internal server error")
        return JSONResponse(content=jsonable_encoder(response), status_code=500)


@app.options("/image")
async def options_image():
    return {"message": "OPTIONS request received. Please use POST method for image upload."}


def initREST(model):
    global network, image_title
    network = model
    uvicorn.run("rest:app", host="0.0.0.0", port=8000, reload=False)
