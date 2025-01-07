# ruff: noqa: I001
"""
This module contains the functions to construct the prompt for the model to
play GeoGuessr and to parse the latitude and longitude from the model response.
"""

from anthropic.types import MessageParam, TextBlock
from anthropic.types.image_block_param import ImageBlockParam, Source

import utils

_LATITUDE_TAG: str = "latitude"
_LONGITUDE_TAG: str = "longitude"

_SYSTEM_PROMPT = """
The assistant is playing GeoGuessr, a game where an image of a random Google \
Street View location is shown and the player has to guess the location of the \
image on a world map.

In the following conversation, the assistant will be shown a single image and \
must make its best guess of the location of the image by providing a latitude \
and longitude coordinate pair.
""".strip()
_HUMAN_PROMPT = f"""
Here is an image from the Geoguessr game.
* Please reason about where you think this image is in <thinking> tags.
* Next, provide your final answer of your predicted latitude and longitude \
coordinates in <{_LATITUDE_TAG}> and <{_LONGITUDE_TAG}> tags.
* The latitude and longitude coordinates that you give me should just be the \
`float` numbers; do not provide any thing else.
* You will NOT be penalized on the length of your reasoning, so feel free to \
think as long as you want.
""".strip()
_ASSISTANT_PREFILL = """
Certainly! I'll analyze the image in <thinking> tags and then provide my \
reasoning and final estimate of the latitude and longitude.

<thinking>
""".strip()


def construct_prompt(image_as_base_64: str) -> tuple[str, list[MessageParam]]:
    """Constructs a prompt for the assistant to play GeoGuessr."""
    image_block = ImageBlockParam(
        source=Source(data=image_as_base_64, media_type="image/jpeg", type="base64"),
        type="image",
    )
    text_block = TextBlock(type="text", text=_HUMAN_PROMPT)
    return (
        _SYSTEM_PROMPT,
        [
            MessageParam(role="user", content=[image_block, text_block]),
            MessageParam(role="assistant", content=_ASSISTANT_PREFILL),
        ],
    )


def parse_latitude_longitude(text: str) -> tuple[float, float] | None:
    """
    Parse the latitude and longitude from the text by finding the first
    occurrence of the XML tags and parsing the string inside.
    """
    latitude_parses = utils.extract_xml_tags(text, _LATITUDE_TAG)
    longitude_parses = utils.extract_xml_tags(text, _LONGITUDE_TAG)

    if not latitude_parses or not longitude_parses:
        return None

    try:
        latitude = float(latitude_parses[0].strip())
        longitude = float(longitude_parses[0].strip())
        return latitude, longitude
    except ValueError as e:
        print(f"Error parsing latitude and longitude: {e}")
        return None
