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
from morph import morph_images

from gradio_client import Client
from similarity_finder.morph_similar_images import get_morphed_images
from inference_script import run

FACEMORPH_API_URL = "https://api.facemorph.me/api"
FACEMORPH_ENCODE_IMAGE = "/encodeimage/"
FACEMORPH_GENERATE_IMAGE = "/face/"

EXPERIMENT_DIR = "experiments/"

client = Client("https://99ashutosh-find-similar-image.hf.space/--replicas/m5fdd/")

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

async def run_test(preset_exp_dir, age, age_range, gender, race, ip2p_prompt):
    # Initial Setup
    exp_dir = preset_exp_dir + "/raw_images"
    os.makedirs(exp_dir)
    print("Made dir(s)")

    # preprocess: facemorph
    await img_format(preset_exp_dir + "/uploaded_image.jpeg", exp_dir)
    print("preprocessed image")
    
    # get similar faces
    similar_images = client.predict(
        f'{exp_dir}/preprocessed_uploaded_image.jpeg',
        age,
        gender,    # str  in 'Gender' Textbox component
        race,        # str  in 'Race' Textbox component
        api_name="/predict"
    )
    similar_images = literal_eval(similar_images)
    similar_images = similar_images[1:6]
    print("similar images: ", similar_images)

    # run morph on top 5
    os.makedirs(preset_exp_dir + "/morphed_images/")
    morphed_images = await get_morphed_images(exp_dir + "/preprocessed_uploaded_image.jpeg", similar_images, exp_dir + "/../morphed_images/")
    print("morphed images: ", morphed_images)

    # run sam
    path1 = exp_dir + "/../SAM_outputs"
    path2 = exp_dir + "/../morphed_images/"
    run(path1, path2, age_range)
    print("ran sam")

    # run GA
    for i in age_range:
        path1 = exp_dir + "/preprocessed_uploaded_image.jpeg"
        path2 = exp_dir + f'/../SAM_outputs/{i}/F0/'
        await morph_images(exp_dir + "/preprocessed_uploaded_image.jpeg", exp_dir + f'/../SAM_outputs/{i}/F0/')
    print("ran GA")

    # run ip2p
    print("ran ip2p")

if __name__ == '__main__':
    preset_exp_dir = "experiments/test-exp-male"
    age = 23
    age_range = ["30", "40", "50", "60", "70"]
    gender = "M"
    race = "White"
    ip2p_prompt = []
    run_test(preset_exp_dir, age, age_range, gender, race, ip2p_prompt)
