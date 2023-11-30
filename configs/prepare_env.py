"""
Script to prepare env before exec
"""
from multiprocessing import Process
import tarfile
import os

MODEL_PATHS = {
    "ffhq_aging": {
        "id": "14NsAD_OQ2xgK5_WEzTvPuJK1rhRpig9G",
        "name": "sam_ffhq_aging.pt",
        "path": "checkpoints"
    },
    "CACD2000": {
        "id": "1JWTqMEiUZ2yNUJJl_5Ctq8SuskVocn51",
        "name": "CACD2000_refined.tar",
        "path": "dataset"
    },
    "insightface-0000": {
        "id": "1-NE3-J7nJIqBVbwKEESpLQ7yFMZtOujR",
        "name": "insightface-0000.params",
        "path": "FaceImageQuality/insightface/model"
    },
    "insightface-symbol": {
        "id": "12CnD7WlYHCb6WCFJOiF8uC8_Q9QoYswa",
        "name": "insightface-symbol.json",
        "path": "FaceImageQuality/insightface/model"
    }
}

"""
,
    "ip2p": {
        "id": "1C8Ka2qGrmERbNEECjljIJshdxitg6xBt",
        "name": "instruct-pix2pix-00-22000.ckpt",
        "path": "ip2p/checkpoints"
    }
"""

def download_file(file_id, file_name, save_path):
    """Function to generate the urls for given params"""
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(
        FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path
    )
    # check_call(['wget', '--load-cookies', '/tmp/cookies.txt', ], stdout=DEVNULL, stderr=subprocess.STDOUT)
    os.system(url)


def download_and_extract_files(file_id, file_name, save_path):
    """Function to generate the urls for given params"""
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(
        FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path
    )
    os.system(url)
    dataset = tarfile.open(save_path + "/" + file_name)
    dataset.extractall(save_path)
    dataset.close()


def download_all_files():
    """
    Function to download all the three files
    """
    programs = []
    for key, details in MODEL_PATHS.items():
        if not os.path.exists(details["path"] + "/" + details["name"]):
            os.makedirs(details["path"], exist_ok=True)
            if (key == "CACD2000" or key == "embeddings"):
                proc = Process(target=download_and_extract_files, args=(
                    details["id"], details["name"], details["path"],))
                programs.append(proc)
                proc.start()
            else:
                proc = Process(target=download_file, args=(
                    details["id"], details["name"], details["path"],))
                programs.append(proc)
                proc.start()

    for proc in programs:
        proc.join()

    print("Environent Ready!")


download_all_files()


"""
def prepareEnv():

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if not os.path.exists("static"):
        os.makedirs("static")

    path = MODEL_PATHS["fittest_dataset"]
    if not os.path.isfile(UPLOAD_FOLDER + "/" + path["name"]):
        downloadFile(path["id"], path["name"], path["path"])

    # download the CACD dataset
    path = MODEL_PATHS["CACD2000"]
    if not os.path.exists(UPLOAD_FOLDER + "/" + 'CACD2000'):
        downloadFile(path["id"], path["name"], path["path"])
        dataset = tarfile.open(path["path"] + "/" + path["name"])
        dataset.extractall(path["path"])
        dataset.close()
"""
