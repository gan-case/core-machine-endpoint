from fastapi import FastAPI, WebSocket
# from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import base64
from ast import literal_eval
import json
import time
import logging
import httpx
import gc
from subprocess import Popen
import uvicorn
import pandas as pd
import shutil

from gradio_client import Client
from similarity_finder.morph_similar_images import get_morphed_images
from inference_script import run

FACEMORPH_API_URL = "https://api.facemorph.me/api"
FACEMORPH_ENCODE_IMAGE = "/encodeimage/"
FACEMORPH_GENERATE_IMAGE = "/face/"

EXPERIMENT_DIR = "experiments/"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client("https://99ashutosh-find-similar-image.hf.space/--replicas/m5fdd/")

test_html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Test</title>
    </head>
    <body>
        <h1>GAN Case Lite</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="file" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };

            async function sendMessage(event) {
                var input = document.getElementById("messageText")
                                  var filereader = new FileReader();
  filereader.readAsDataURL(input.files[0]);
  filereader.onload = function (evt) {
     var base64 = evt.target.result;
    ws.send(base64)
  }
//                 input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""

async def img_format(image, uuid):
    data = {'usrimg': open(image, 'rb')}
    j = {'tryalign': True}

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(FACEMORPH_API_URL +
                              FACEMORPH_ENCODE_IMAGE, files=data, data=j)
        
        rjson = json.loads(r.text)
        values = {'guid': rjson['guid']}
    
        r = await client.get(FACEMORPH_API_URL + FACEMORPH_GENERATE_IMAGE, params=values)
        rawimg = r.content
        with open(uuid + "/preprocessed_uploaded_image.jpeg", 'wb') as out_file:
            out_file.write(rawimg)

@app.get("/")
async def get():
    #return FileResponse("index.html")
    return "ok"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()

        # Initial Setup
        exp_uuid = str(uuid.uuid4())
        exp_dir = EXPERIMENT_DIR + "complete-example"  #exp_uuid
        raw_images_location = exp_dir + "/raw_images"
        os.makedirs(raw_images_location)
        await websocket.send_json({"status_code": 1, "exp_uuid": exp_uuid})

        # Process uploaded image
        uploaded_image = open(exp_dir + "/uploaded_image.jpeg", "wb")
        tmp = data
        data = data.replace('data:image/jpeg;base64,', '')
        uploaded_image.write(base64.b64decode(data))
        uploaded_image.close()
        await websocket.send_json({"status_code": 2, "exp_uuid": exp_uuid, "image": tmp})

        # preprocess: facemorph
        await img_format(exp_dir + "/uploaded_image.jpeg", raw_images_location)
        preprocessed_file = open(raw_images_location + "/preprocessed_uploaded_image.jpeg", "rb").read()
        base64_utf8_str = base64.b64encode(preprocessed_file).decode('utf-8')
        dataurl = f'data:image/jpeg;base64,{base64_utf8_str}'
        await websocket.send_json({"status_code": 3, "exp_uuid": exp_uuid, "image": dataurl})

        # get similar faces
        similar_images = client.predict(
            f'{raw_images_location}/preprocessed_uploaded_image.jpeg',
            "23",
            "M",    # str  in 'Gender' Textbox component
            "Male",        # str  in 'Race' Textbox component
            api_name="/predict"
        )
        similar_images = literal_eval(similar_images)
        similar_images = similar_images[1:6]
        similar_images_encoded = []
        for image in similar_images:
            similar_image_data = open("dataset/CACD2000/" + image, "rb").read()
            shutil.copyfile("dataset/CACD2000/" + image, exp_dir +"/"+ image)
            base64_utf8_str = base64.b64encode(similar_image_data).decode('utf-8')
            dataurl = f'data:image/jpeg;base64,{base64_utf8_str}'
            similar_images_encoded.append({"imagename": image, "encodedstring": dataurl})      
        await websocket.send_json({"status_code": 4, "exp_uuid": exp_uuid, "images": similar_images_encoded})

        # run morph on top 5
        morphed_images_location = exp_dir + "/morphed_images/"
        os.makedirs(morphed_images_location)
        morphed_images = await get_morphed_images(raw_images_location + "/preprocessed_uploaded_image.jpeg", similar_images, morphed_images_location)

        # run sam
        path1 = exp_dir + "/SAM_outputs"
        age = ["70"]
        run(path1, morphed_images_location, age)
        #sam_output_location = exp_dir + "/SAM_outputs/70/F0/"
        #sam_processed_images = os.listdir(sam_output_location)
        #sam_processed_encoded = []
        #for image in sam_processed_images:
        #    sam_processed_data = open(sam_output_location + image, "rb").read()
        #    base64_utf8_str = base64.b64encode(sam_processed_data).decode('utf-8')
        #    dataurl = f'data:image/jpeg;base64,{base64_utf8_str}'
        #    sam_processed_encoded.append({"imagename": image, "encodedstring": dataurl})      
        #await websocket.send_json({"status_code": 6, "exp_uuid": exp_uuid, "images": sam_processed_encoded})

        time.sleep(20)
        # run GA
        #for i in ["30", "40", "50", "60", "70"]:
        #    path1 = exp_dir + "/preprocessed_uploaded_image.jpeg"
        #    path2 = exp_dir + f'/../SAM_outputs/{i}/F0/'
        #    await morph_images(exp_dir + "/preprocessed_uploaded_image.jpeg", exp_dir + f'/../SAM_outputs/{i}/F0/')
        ga_outputs = exp_dir + "/ga-outputs"
        ga_outputs_encoded = []

        count = 0
        for generation in os.listdir(ga_outputs):
            ga_outputs_encoded.append({"generation": generation, "images": []})
            for image in os.listdir(ga_outputs + "/" + generation):
                ga_processed_data = open(ga_outputs + "/" + generation + "/" + image, "rb").read()
                base64_utf8_str = base64.b64encode(ga_processed_data).decode('utf-8')
                dataurl = f'data:image/jpeg;base64,{base64_utf8_str}'
                ga_outputs_encoded[count]['images'].append({"imagename": image, "encodedstring": dataurl})
            count += 1
        await websocket.send_json({"status_code": 7, "exp_uuid": exp_uuid, "images": ga_outputs_encoded})

        # Completed

        await websocket.send_json({"status_code": 8, "exp_uuid": exp_uuid})
        

if __name__ == '__main__':
    #download_all_files()
    uvicorn.run("main:app")
