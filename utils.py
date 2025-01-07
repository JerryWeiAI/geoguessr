import io
import json
import math
import os
import random
import re
from base64 import standard_b64encode
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Literal

import jsonlines
import numpy as np
import orjson
import pandas as pd
from PIL import Image
from tqdm import tqdm

MAX_SCORE: Literal[5000] = 5000


@dataclass
class Coordinate:
    """Stores a pair of latitude and longitude coordinates."""

    latitude: float
    longitude: float

    def distance(self, other: "Coordinate") -> float:
        """
        Calculates the distance between two coordinates in kilometers by
        using the haversine formula.
        """
        R = 6371  # Earth's radius in kilometers

        # Convert latitude and longitude from degrees to radians
        lat1 = math.radians(self.latitude)
        lon1 = math.radians(self.longitude)
        lat2 = math.radians(other.latitude)
        lon2 = math.radians(other.longitude)

        # Calculate differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Apply haversine formula
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        # Calculate the distance
        return R * c


def score(distance: float) -> int:
    """
    Calculates the geoguessr score based on the distance of the guess from the
    actual location (in km). Assumes that you're playing on a world map.

    Formula taken from https://osv5m-plonk.hf.space/.
    """
    return round(5000 * np.exp(-distance / 1492.7))


def extract_xml_tags(text: str, tag: str) -> list[str]:
    """Extracts the contents of all XML tags with the given tag."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def serialize_img(img: Image.Image, target_format: str = "jpeg") -> str:
    """Convert image to base64 encoded string."""
    f = io.BytesIO()
    img.save(f, format=target_format, lossless=target_format != "jpeg")
    image_bytes = f.getvalue()
    return standard_b64encode(s=image_bytes).decode()


def jsonl_save_dataframe(
    path: str,
    df: pd.DataFrame,
    overwrite: bool = True,
    chunk_size: int = 100,
    disable_tqdm: bool = False,
) -> None:
    """Save DataFrame to JSONL file with memory-efficient chunking."""
    assert overwrite or not os.path.exists(path), "File already exists!"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    total_rows = len(df)
    chunks = list(range(0, total_rows, chunk_size))

    with open(path, mode="w") as f:
        w = jsonlines.Writer(f)

        for start in tqdm(chunks, disable=disable_tqdm, desc=f"jsonl_save_dataframe"):
            chunk = df.iloc[start : start + chunk_size]

            for record in chunk.to_dict("records"):
                try:
                    json.dumps(record)
                    w.write(record)
                except (TypeError, OverflowError):
                    fallback_json = pd.DataFrame([record]).to_json(orient="records")

                    if fallback_json is not None:
                        w.write(json.loads(fallback_json)[0])
                    else:
                        w.write({})


def jsonl_load(path: str) -> Generator[dict, None, None]:
    """Load a JSONL file."""
    with open(path) as file:
        for line in file:
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError:
                yield json.loads(line)


def jsonl_load_dataframe(path: str, disable_tqdm: bool = True) -> pd.DataFrame:
    """Load a DataFrame from a JSONL file."""
    return pd.DataFrame(
        list(tqdm(jsonl_load(path), disable=disable_tqdm, desc="jsonl_list"))
    )


def listdir(path: str) -> list[str]:
    """List all non-hidden files in the given directory."""
    return [os.path.join(path, f) for f in os.listdir(path) if not path.startswith(".")]


def choose_items(items: list[Any], n: int, rng: random.Random) -> list[Any]:
    """Chooses `n` items from `items` using `rng`."""
    if n > 0 and n <= len(items):
        return rng.sample(items, n)
    else:
        return items
