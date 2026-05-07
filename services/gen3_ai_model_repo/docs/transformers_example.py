# clear cache: rm -r ~/.cache/gen3
# run with: HF_HUB_CACHE=~/.cache/gen3 HF_TOKEN=foobar HF_ENDPOINT=http://0.0.0.0:4141 uv run python transformers_example.py

import os

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME") or "uc-ctds/bge-large-en-v1.5-bio-mapping"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# quick test: encode a sample sentence
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
