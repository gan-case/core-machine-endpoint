from FaceImageQuality.face_image_quality import SER_FIQ
import os
import argparse
import cv2
import requests
import json
import random
import httpx
import logging
from annoy import AnnoyIndex
from deepface import DeepFace
import json
import asyncio

def image_to_embedding(image_path):
    try:
        embedding_json = {}
        embedding_json['image_name'] = image_path
        embedding_objs = DeepFace.represent(img_path=image_path)
        embedding_json.update(embedding_objs[0])
        return embedding_json
    except Exception as e:
        print("Error at " + image_path)
        print(e)
        return None


def calculate_similarity_scores(target_embedding, other_embeddings, n_neighbors=10):
    f = len(target_embedding)
    t = AnnoyIndex(f, metric='euclidean')
    ntree = 50

    for i, vector in enumerate(other_embeddings):
        t.add_item(i, vector)
    t.build(ntree)

    similar_img_ids, distances = t.get_nns_by_vector(target_embedding, n_neighbors, include_distances=True)
    return similar_img_ids, distances


def process_and_calculate_similarity(target_image_path, other_image_paths):
    target_embedding = image_to_embedding(target_image_path)
    if target_embedding is None:
        return None

    other_embeddings = [image_to_embedding(image_path) for image_path in other_image_paths if image_path != target_image_path]
    other_embeddings = [emb for emb in other_embeddings if emb is not None]

    if len(other_embeddings) < 10:
        print("Error: Could not process all 10 images.")
        return None

    similar_img_ids, distances = calculate_similarity_scores(target_embedding['embedding'], [emb['embedding'] for emb in other_embeddings], len(other_embeddings))

    return distances

def fitness_fun(path,sim_dict,ser_fiq):
    test_img = cv2.imread(path)
    aligned_img = ser_fiq.apply_mtcnn(test_img)
    score = 1 - 0.4*ser_fiq.get_score(aligned_img, T=100)+0.6*sim_dict[path]
    logging.info(path,score)
    return(score)

async def morph_images(image_to_morph, f0_location):
    ser_fiq = SER_FIQ(gpu=0)
    upload_url = "https://api.facemorph.me/api/encodeimage/"
    morph_url = "https://api.facemorph.me/api/morphframe/"
    current_gen = 0
    total_gen = 4
    fit_check_files = os.listdir(f0_location)
    fit_check_files = list(map(lambda x: f0_location + x, fit_check_files))
    sim_dict = dict()

    while(current_gen != total_gen):
        logging.info("Generation:",current_gen)
        # PHASE 1: READ THE IMAGE NAMES IN THE DIRECTORY
        I_list = fit_check_files

        # PHASE 2: UPLOAD THE IMAGES TO THE SERVER AND GET THE REFERENCE GUID
        guid_dict = dict()
        for file in I_list:
            upload_flags = {
                "tryalign":False
                }
            upload_data = {"usrimg":open(file,'rb')}
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(upload_url, files=upload_data, data=upload_flags)
                json_response = json.loads(response.text)['guid']
                guid_dict[file] = json_response

        # PHASE 3: MORPH THE IMAGES AND SAVE THE OUTPUT
        current_gen += 1
        os.makedirs(f0_location+'../F'+str(current_gen))
        morph_file_input = list(guid_dict.keys())
        for file1 in range(len(morph_file_input)):
            for file2 in range(file1+1,len(morph_file_input)):
                f_num=random.randint(0,51)
                morph_flags = {
                    "from_guid":guid_dict[morph_file_input[file1]],
                    "to_guid":guid_dict[morph_file_input[file2]],
                    "frame_num":f_num,
                    "linear":True
                    }
                async with httpx.AsyncClient(timeout=None) as client:
                    morph_response = await client.get(morph_url, params=morph_flags)
                    with open(f0_location+'../F'+str(current_gen)+'/'+str(morph_file_input[file1].split('/')[-1].split('.')[0])+"_"+str(morph_file_input[file2].split('/')[-1]),"wb") as morphed_file:
                        morphed_file.write(morph_response.content)
                        fit_check_files.append(f0_location+'../F'+str(current_gen)+'/'+str(morph_file_input[file1].split('/')[-1].split('.')[0])+"_"+str(morph_file_input[file2].split('/')[-1]))
        
        # PHASE 4: TAKE MORPHED IMAGES AND RUN FITNESS FUNCTION ON IT
        sim_vec = process_and_calculate_similarity(image_to_morph,fit_check_files)
        logging.info(fit_check_files)
        for i in range(len(sim_vec)):
          sim_dict[fit_check_files[i]] = sim_vec[i]
        fit_check_files = sorted(fit_check_files,key=lambda x:fitness_fun(x,sim_dict,ser_fiq))
        logging.info(fit_check_files)
        fit_check_files = fit_check_files[:5]
        logging.info(fit_check_files)
        

if __name__ == '__main__':
	#parser = argparse.ArgumentParser(description='Process some integers.')
	#parser.add_argument("path1")
	#parser.add_argument("path2")
	#parser.add_argument("age")

	#args = parser.parse_args()
	#run(args.path1, args.path2)


    loop = asyncio.get_event_loop()
    loop.run_until_complete(morph_images("experiments/8f749586-c9a3-4f0a-b7a5-dd8678353dda/raw_images/preprocessed_uploaded_image.jpeg","experiments/8f749586-c9a3-4f0a-b7a5-dd8678353dda/SAM_outputs/30/F0/"))