# clear cache: rm -r ~/.cache/gen3
# run with: HF_HUB_CACHE=~/.cache/gen3 HF_TOKEN=foobar HF_ENDPOINT=http://0.0.0.0:4141 uv run python transformers_example.py


from huggingface_hub import hf_hub_download

HF_ENDPOINT = "http://127.0.0.1:4141"
# from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "test/repo"
# or "uc-ctds/bge-large-en-v1.5-bio-mapping"

path = hf_hub_download(repo_id=MODEL_NAME, filename="config.json", revision="main", endpoint=HF_ENDPOINT)
# outputs = model(**inputs)
#
# print(outputs.last_hidden_state.shape)
print(f"Downloaded file to: {path}")

with open(path) as f:
    print("\nFile contents:")
    print(f.read())
