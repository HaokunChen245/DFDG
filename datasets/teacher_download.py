import argparse
import os
import tarfile
from zipfile import ZipFile

import gdown


def download_and_extract(url, dst, remove=True):
    if "drive.google" in url:
        gdown.download(url, dst, quiet=False)

        if dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(os.path.dirname(dst))
            tar.close()

        if dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(os.path.dirname(dst))
            tar.close()

        if dst.endswith(".zip"):
            zf = ZipFile(dst, "r")
            zf.extractall(os.path.dirname(dst))
            zf.close()

    else:
        import time

        import requests

        chunk_size = 1024 * 1024
        begin = time.time()

        session = requests.Session()
        req = session.get(url, stream=True)
        with open(dst, "w") as f:
            for chunk in req.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()

        end = time.time()
        print(
            f" downloaded {dst} in {(end - begin)/60} minutes, with chunk size = {chunk_size}"
        )

        if dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(os.path.dirname(dst))
            tar.close()

        if dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(os.path.dirname(dst))
            tar.close()

        if dst.endswith(".zip"):
            zf = ZipFile(dst, "r")
            zf.extractall(os.path.dirname(dst))
            zf.close()

    if remove:
        os.remove(dst)

def download_teacher(root_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = os.path.join(root_dir, "saved_models_teacher")

    if os.path.exists(full_path):
        print("data folder exists already")
        if len(os.listdir(full_path)) == 0:
            print("data folder empty")
            download_and_extract(
                "https://drive.google.com/uc?id=1vUG3QGr5Wfj8aguddgeaI2S-LkzniCz5",
                os.path.join(root_dir, "teacher_models.zip"),
            )

            os.rename(os.path.join(
                root_dir, "saved_models_teacher"), full_path)

    else:
        download_and_extract(
            "https://drive.google.com/uc?id=1vUG3QGr5Wfj8aguddgeaI2S-LkzniCz5",
            os.path.join(root_dir, "teacher_models.zip"),
        )

        os.rename(os.path.join(root_dir, "saved_models_teacher"), full_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained teacher models")
    parser.add_argument("--root_dir", type=str, required=True)

    args = parser.parse_args()

    if len(args.root_dir)>0 and not os.path.isdir(args.root_dir):
        os.system(f'mkdir -p {args.root_dir}')

    download_teacher(args.root_dir)
