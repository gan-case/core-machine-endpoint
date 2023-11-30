import json
import random
import httpx
import json

upload_url = "https://api.facemorph.me/api/encodeimage/"
morph_url = "https://api.facemorph.me/api/morphframe/"

async def get_morphed_images(original_image, images_to_morph_with, output_path):
    full_images_to_morph_with = ["dataset/CACD2000/" + image for image in images_to_morph_with]
    input_images = [original_image] + full_images_to_morph_with
    morphed_images = []

    # PHASE 2: UPLOAD THE IMAGES TO THE SERVER AND GET THE REFERENCE GUID
    guid_dict = dict()
    for file in input_images:
        upload_flags = {
            "tryalign":False
        }
        upload_data = {"usrimg":open(file,'rb')}
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(upload_url, files=upload_data, data=upload_flags)
            json_response = json.loads(response.text)['guid']
            guid_dict[file] = json_response

    # PHASE 3: MORPH THE IMAGES AND SAVE THE OUTPUT
    morph_file_input = list(guid_dict.keys())
    for file1 in range(len(images_to_morph_with)):
        f_num=random.randint(0,51)
        morph_flags = {
            "from_guid":guid_dict[morph_file_input[0]],
            "to_guid":guid_dict[morph_file_input[file1]],
            "frame_num":f_num,
            "linear":True
        }
        async with httpx.AsyncClient(timeout=None) as client:
            morph_response = await client.get(morph_url, params=morph_flags)
            with open(output_path + images_to_morph_with[file1],"wb") as morphed_file:
                morphed_file.write(morph_response.content)
                morphed_images.append(images_to_morph_with[file1])

    return morphed_images