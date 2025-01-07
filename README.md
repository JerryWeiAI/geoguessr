# Claude plays GeoGuessr

This repository contains a simple setup for downloading Google Street View images and prompting Claude to predict the latitude/longitude of the image to simulate [GeoGuessr](https://www.geoguessr.com/).

## Setup

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Playing GeoGuessr

1. Download the [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) dataset from HuggingFace and preprocess the data by running this command:

```bash
python -m data_process
```

2. Set your Anthropic API Key (instructions for getting an API key can be found [here](https://docs.anthropic.com/en/api/getting-started)) using:

```bash
export ANTHROPIC_API_KEY=<api_key>
```

3. Select a [Claude model to use](https://docs.anthropic.com/en/docs/about-claude/models) (e.g., `claude-3-5-sonnet-20241022`) and run the evaluation script using:

```bash
python -m play \
    --model="claude-3-5-sonnet-20241022" \
    --num-examples=1000
```

4. The evaluation script will print out the max, min, average, and median score of the examples that were used.

## Citing this work

If you find this code useful, feel free to cite:

```
@misc{jerry2025claude,
  author = {Jerry Wei},
  title = {Claude plays GeoGuessr},
  date = {2025-01-06},
  year = {2025},
  url = {https://www.jerrywei.net/blog/claude-plays-geoguessr},
}
```

# License

MIT License

Copyright (c) 2025 Jerry Wei.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
