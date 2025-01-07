# ruff: noqa: I001
"""Play GeoGuessr with a model and calculate metrics."""

import argparse
import concurrent.futures
import os
import random
from dataclasses import dataclass
from textwrap import dedent

import anthropic
import pandas as pd
import trio
from anthropic.types import MessageParam

import data_process
import prompting
import utils

_RANDOM_SEED: int = 42


@dataclass
class Example:
    """Class to store data for a single GeoGuessr example."""

    image_b64: str
    target_coords: utils.Coordinate
    predicted_coords: utils.Coordinate | None = None
    raw_model_response: str | None = None

    def score(self) -> float:
        """Calculate the GeoGuessr score for this example."""
        if self.predicted_coords is None:
            return 0

        return utils.score(self.target_coords.distance(self.predicted_coords))

    def to_dict(self) -> dict[str, str | float | None]:
        """Convert the example to a dictionary."""
        return {
            "image_b64": self.image_b64,
            "target_latitude": self.target_coords.latitude,
            "target_longitude": self.target_coords.longitude,
            "predicted_latitude": (
                self.predicted_coords.latitude
                if self.predicted_coords is not None
                else None
            ),
            "predicted_longitude": (
                self.predicted_coords.longitude
                if self.predicted_coords is not None
                else None
            ),
            "raw_model_response": self.raw_model_response,
            "score": self.score(),
        }


def process_file(file: str) -> list[Example]:
    """Process a single file and return a list of Examples."""
    result = []
    in_df = utils.jsonl_load_dataframe(file)
    print(f"Loading examples from {file}...")

    for _, row in in_df.iterrows():
        result.append(
            Example(
                image_b64=row[data_process.B64_IMAGE_KEY],
                target_coords=utils.Coordinate(
                    latitude=row[data_process.LATITUDE_KEY],
                    longitude=row[data_process.LONGITUDE_KEY],
                ),
            )
        )

    return result


def load_examples_from_metadata(metadata_path: str) -> list[Example]:
    """Load all examples from the given metadata file using multithreading."""
    all_files = utils.listdir(metadata_path)
    result: list[Example] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as e:
        future_to_file = {e.submit(process_file, file): file for file in all_files}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result.extend(future.result())
            except Exception as exc:
                print(f"{file} generated an exception: {exc}")

    print(f"Loaded {len(result)} examples from {metadata_path}")
    return result


def get_model_response(
    prompt: tuple[str, list[MessageParam]], model: str, client: anthropic.Anthropic
) -> str:
    """Get the model response for the given prompt."""
    system, messages = prompt
    result = client.messages.create(
        model=model,
        max_tokens=4095,
        system=system,
        messages=messages,
    )
    return result.content[0].text


async def get_model_responses(
    examples: list[Example], model: str, client: anthropic.Anthropic
) -> list[Example]:
    """
    Prompts the model to play GeoGuessr on the given examples, parses the predicted
    coordinates, and returns the updated examples.
    """
    result: list[Example] = []
    prompts = [prompting.construct_prompt(ex.image_b64) for ex in examples]
    print(f"-----Sample prompt-----\n{prompts[0]}\n-----------------------")
    responses = [get_model_response(prompt, model, client) for prompt in prompts]
    print(f"-----Sample response-----\n{responses[0]}\n-----------------------")

    for example, response in zip(examples, responses, strict=True):
        example.raw_model_response = response
        parsed_response = prompting.parse_latitude_longitude(response)

        if parsed_response is not None:
            example.predicted_coords = utils.Coordinate(
                latitude=parsed_response[0], longitude=parsed_response[1]
            )

        result.append(example)

    return result


async def run_experiment(
    model: str,
    num_examples: int,
    client: anthropic.Anthropic,
    metadata_path: str,
) -> pd.DataFrame:
    """Run the experiment with the given model."""
    in_examples: list[Example] = load_examples_from_metadata(metadata_path)
    in_examples = utils.choose_items(
        in_examples, n=num_examples, rng=random.Random(_RANDOM_SEED)
    )
    print(f"Sampling {len(in_examples)} examples...")
    print(f"Beginning sampling with model {model}...")
    out_examples = await get_model_responses(in_examples, model, client)
    print(f"Finished sampling {len(out_examples)} examples.")
    return pd.DataFrame([ex.to_dict() for ex in out_examples])


def calculate_metrics(df: pd.DataFrame, model: str) -> str:
    """Calculate the metrics for the given DataFrame."""
    return dedent(f"""\
        Metrics:
        Max score: `{df['score'].max()}`
        Min score: `{df['score'].min()}`
        Average score: `{df['score'].mean()}`
        Median score: `{df['score'].median()}`
    """)


def main(*, model: str, num_examples: int) -> None:
    """Main function to run the experiment."""
    # Check if the API key can be set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("Please set the API key using `ANTHROPIC_API_KEY=<key>`!")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    in_folder = "/".join(data_process.OUT_JSONL_PATH.split("/")[:-1])
    out_df = trio.run(run_experiment, model, num_examples, client, in_folder)
    out_df = out_df.dropna(subset=["predicted_latitude", "predicted_longitude"])
    metrics_message = calculate_metrics(out_df, model)
    print(f"GeoGuessr experiment completed for {model}!")
    print(metrics_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model to use for GeoGuessr.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        required=True,
        help="The number of examples to sample.",
    )
    args = parser.parse_args()
    main(model=args.model, num_examples=args.num_examples)


"""
Example run command:
```
python -m play \
    --model="claude-3-5-sonnet-20241022" \
    --num-examples=10
```
"""
