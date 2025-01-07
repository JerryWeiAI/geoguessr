# ruff: noqa: I001

import multiprocessing
import os
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

import utils

OUT_JSONL_PATH: str = "geoguessr/data.jsonl"

ID_KEY: str = "image_name"
LATITUDE_KEY: str = "latitude"
LONGITUDE_KEY: str = "longitude"
B64_IMAGE_KEY: str = "image_b64"

_DATA_FOLDER: str = "OpenWorld"
_BATCH_SIZE: int = 10_000


def process_image(args: tuple[str, float, float]) -> dict[str, str | float]:
    """
    Process a single image: copy it to the output path and create a data point.
    """
    image, latitude, longitude = args
    loaded_image = Image.open(image)
    base_64_image = utils.serialize_img(loaded_image, target_format="jpeg")
    return {
        ID_KEY: image.split("/")[-1],
        LATITUDE_KEY: latitude,
        LONGITUDE_KEY: longitude,
        B64_IMAGE_KEY: base_64_image,
    }


def load_metadata_key(metadata_path: str) -> dict[str, tuple[float, float]]:
    """Load the latitude and longitude from the metadata file."""
    with open(metadata_path) as f:
        df = pd.read_csv(f, sep=",")

    ids: list[str] = [str(id) for id in df["id"].tolist()]
    latitudes: list[float] = [float(l) for l in df["latitude"].tolist()]
    longitudes: list[float] = [float(l) for l in df["longitude"].tolist()]
    return {ids[i]: (latitudes[i], longitudes[i]) for i in range(len(ids))}


def save_batch(batch: list[dict[str, str | float]], batch_number: int) -> None:
    """
    Save a batch of processed data to a JSONL file.
    """
    batch_df = pd.DataFrame(batch)
    batch_path = OUT_JSONL_PATH.replace(".jsonl", f"-part{batch_number:04d}.jsonl")
    utils.jsonl_save_dataframe(batch_path, batch_df, disable_tqdm=True)
    print(f"Saved batch {batch_number} to: {batch_path}")


def main(folder: str, metadata_path: str) -> None:
    """
    Main function to process the images in the input folder using multiprocessing.
    """
    folders = utils.listdir(folder)
    metadata = load_metadata_key(metadata_path)

    # Prepare the arguments for multiprocessing
    process_args: list[tuple[str, float, float]] = []
    total_images: int = 0

    for i, curr_folder in enumerate(folders):
        folder_images = utils.listdir(curr_folder)
        total_images += len(folder_images)

        for image in folder_images:
            image_name = image.split("/")[-1]
            image_id = (
                image_name[: -len(".jpg")]
                if image_name.endswith(".jpg")
                else image_name
            )
            latitude, longitude = metadata[image_id]
            process_args.append((image, latitude, longitude))

        print(f"Prepared folder {i + 1} / {len(folders)}")

    print(f"Total images to process: {total_images}")

    # Use multiprocessing to process images with progress bar
    with multiprocessing.Pool(processes=32) as pool:
        batch: list[dict[str, str | float]] = []
        batch_number: int = 1

        for result in tqdm(
            pool.imap(process_image, process_args),
            total=total_images,
            desc="Processing images",
        ):
            batch.append(result)

            if len(batch) >= _BATCH_SIZE:
                save_batch(batch, batch_number)
                batch = []
                batch_number += 1

        # Save any remaining data in the last batch
        if batch:
            save_batch(batch, batch_number)

    print(
        f"Completed processing and saving {total_images} images in {batch_number} batches."
    )


def download_data(images_dir: str) -> None:
    """
    Download the GeoGuessr dataset from HuggingFace.

    Nested structure should be:
    ```
    _DATA_FOLDER
    ├── data
    │   ├── images
    │   │   └── test
    │   │       ├── 00.zip
    │   │       ├── 01.zip
    │   │       ├── 02.zip
    │   │       ├── 03.zip
    │   │       └── 04.zip
    ├── images
    │   └── test
    │       ├── 00
    │       │   ├── *.jpg
    │       ├── 01
    │       │   ├── *.jpg
    │       ├── 02
    │       │   ├── *.jpg
    │       ├── 03
    │       │   ├── *.jpg
    │       └── 04
    │           ├── *.jpg
    ├── test.csv
    ```
    """
    for i in range(5):
        hf_hub_download(
            repo_id="osv5m/osv5m",
            filename=str(i).zfill(2) + ".zip",
            subfolder="images/test",
            repo_type="dataset",
            local_dir=os.path.join(_DATA_FOLDER, "data"),
        )

    for path in utils.listdir(os.path.join(_DATA_FOLDER, "data/images/test/")):
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(images_dir)

    hf_hub_download(
        repo_id="osv5m/osv5m",
        filename="test.csv",
        repo_type="dataset",
        local_dir=_DATA_FOLDER,
    )


if __name__ == "__main__":
    images_dir = os.path.join(_DATA_FOLDER, "images/test/")
    metadata_path = os.path.join(_DATA_FOLDER, "test.csv")

    if not os.path.exists(images_dir) or not os.path.exists(metadata_path):
        download_data(images_dir)

    main(images_dir, metadata_path)
