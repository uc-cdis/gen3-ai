# Hugging Face - Compatible API Proof of Concept

This is a very very rough, mocked proof-of-concept for an API that mimics the minimal behavior of the [HuggingFace Hub](https://huggingface.co/) so that HF clients can use it for model download. This mock PoC uses local files exposed through a FastAPI serve with a ton of hard-coding fake values.

If you run this mock server and override the HF domain and tokens, you can see the default huggingface_hub, transformers, and the HF CLI still works without other modifications.

## Setup

Clone our embedding model from HF into `./hf_api/testfiles/`. It's here:

https://huggingface.co/uc-ctds/bge-large-en-v1.5-bio-mapping/tree/main

Now setup a Python env.

```bash
virtualenv .venv
source .venv/bin/activate
pip install huggingface_hub fastapi uvicorn transformers torch
```

Create `hf_api.py`:

```python
import hashlib
from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Response, HTTPException, status, Query
from fastapi.responses import RedirectResponse, StreamingResponse

app = FastAPI(title="HF-Mock API (filesystem-backed)")

BASE_FILES_DIR = Path(__file__).parent / "hf_api" / "testfiles"
FAKE_COMMIT = "abcdef1234567890"
FAKE_ETAG = "etag-123456"


@app.get("/api/models/{namespace}/{repo}/tree/{rev}")
@app.get("/api/models/{namespace}/{repo}/tree/{rev}/{path:path}")
async def list_repo_tree(
    namespace: str,
    repo: str,
    rev: str,
    path: str = "",
    expand: bool = Query(
        False, description="return commit data & minimal security info"
    ),
):
    """
    Return a flat list of entries for the directory *path* (or the file
    itself).  The output matches the structure documented by Hugging Face
    but contains only the essential fields.
    """
    target = BASE_FILES_DIR / Path(path)

    # validate the path exists
    if not target.exists():
        raise HTTPException(status_code=404, detail="Folder not found")

    # gather all files under target. If target is a file, return it
    # as the single entry
    if target.is_file():
        files = [target]
    else:
        files = [path for path in target.rglob("*") if path.is_file()]

    def make_entry(path: Path) -> Dict[str, Any]:
        rel = str(path.relative_to(BASE_FILES_DIR))
        oid = FAKE_COMMIT
        size = path.stat().st_size
        return {
            "type": "file",
            "oid": oid,
            "size": size,
            "path": rel,
            "lfs": {"oid": oid, "size": size, "pointerSize": size},
            "xetHash": None,
            "lastCommit": {
                "id": oid,
                "title": "Mock title",
                "date": datetime.now().isoformat(timespec="seconds") + "Z",
            },
            "securityFileStatus": {
                "status": "unscanned",
                "jFrogScan": {"status": "unscanned"},
                "protectAiScan": {"status": "unscanned"},
                "avScan": {"status": "unscanned"},
                "pickleImportScan": {"status": "unscanned"},
                "virusTotalScan": {"status": "unscanned"},
            },
        }

    return [make_entry(path) for path in files]


@app.get("/api/models/{namespace}/{repo}/revision/{rev}")
async def get_revision(namespace: str, repo: str, rev: str):
    return {
        "id": f"{namespace}/{repo}",
        "revision": rev,
        "sha": FAKE_COMMIT,
        "commit": FAKE_COMMIT,
        "tags": ["latest", "main"],
    }


@app.head("/{namespace}/{repo}/resolve/{rev}/{path:path}")
async def head_file(namespace: str, repo: str, rev: str, path: str):
    local_path = _get_local_file(path.split("/"))
    content = _read_file(local_path)

    size = len(content)
    commit_hash, etag = _compute_hashes(content)

    # also mock the redirected signed URL locally via this same
    # web service. this will stream the file contents as if it
    # was a signed URL
    signed_url = urljoin(
        f"http://0.0.0.0:4099/signed-url/",
        path,
    )

    headers = {
        "X-Repo-Commit": commit_hash,
        "X-Linked-Etag": etag,
        "X-Linked-Size": str(size),
        "Location": signed_url,
    }
    return Response(status_code=status.HTTP_200_OK, headers=headers)


@app.get("/{namespace}/{repo}/resolve/{rev}/{path:path}")
async def get_file(namespace: str, repo: str, rev: str, path: str):
    signed_url = urljoin(
        f"http://0.0.0.0:4099/signed-url/",
        path,
    )
    # this redirect is how our service would work. we'd do auth checks, find
    # the file in s3, create a signed URL and return
    return RedirectResponse(url=signed_url, status_code=status.HTTP_302_FOUND)


@app.get("/signed-url/{path:path}")
async def signed_url(path: str):
    """
    Return the file content as a streaming response.
    This is necessary for large files and guarantees the
    client sees a proper `Content-Length` header.
    """
    local_path = _get_local_file(path.split("/"))
    file_size = local_path.stat().st_size

    media_type = (
        "application/json" if path.endswith(".json") else "application/octet-stream"
    )

    # yields the file in chunks
    def file_iterator(path: Path, chunk_size: int = 65536):
        with path.open("rb") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    headers = {
        "Content-Length": str(file_size),
        "Content-Type": media_type,
    }

    return StreamingResponse(
        file_iterator(local_path),
        media_type=media_type,
        headers=headers,
    )


def _get_local_file(path_parts: list[str]) -> Path:
    local_path = BASE_FILES_DIR.joinpath(*path_parts)
    if not local_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return local_path


def _read_file(local_path: Path) -> bytes:
    return local_path.read_bytes()


def _compute_hashes(content: bytes) -> tuple[str, str]:
    commit_hash = hashlib.sha256(content).hexdigest()
    etag = hashlib.md5(content).hexdigest()
    return commit_hash, etag
```

Run the FastAPI server: `uvicorn hf_api:app --host 0.0.0.0 --port 4099 --reload`

Now create `test_hf.py`:

```python
# clear cache: rm -r ~/.cache/gen3
# run with: HF_HUB_CACHE=~/.cache/gen3 HF_TOKEN=foobar HF_ENDPOINT=http://0.0.0.0:4099 python test_hf.py

### HF CLI

# HF_HUB_CACHE=~/.cache/gen3 HF_TOKEN=foobar HF_ENDPOINT=http://0.0.0.0:4099 hf download uc-ctds/bge-large-en-v1.5-bio-mapping

### HF Python Library
print("------------------- HF Python Library test starting... -------------------")
# pip install huggingface_hub

from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
    upload_folder,
    upload_file,
)

print("------------------- HF Python Library test one file... -------------------")
config_path = hf_hub_download(
    repo_id="uc-ctds/bge-large-en-v1.5-bio-mapping", filename="config.json"
)
print(config_path)

print("------------------- HF Python Library test ALL files... -------------------")
snapshot_download(repo_id="uc-ctds/bge-large-en-v1.5-bio-mapping")

### Transformers Library
print("------------------- transformers test starting... -------------------")
# Note: This USES huggingface_hub, particularly the code is:
"""
.venv/lib/python3.13/site-packages/transformers/utils/hub.py
...
    user_agent = http_user_agent(user_agent)
    # download the files if needed
    try:
        if len(full_filenames) == 1:
            # This is slightly better for only 1 file
            hf_hub_download(
                path_or_repo_id,
                filenames[0],
                subfolder=None if len(subfolder) == 0 else subfolder,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                user_agent=user_agent,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )
        else:
            snapshot_download(
                path_or_repo_id,
                allow_patterns=full_filenames,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                user_agent=user_agent,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )
...
"""

from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "uc-ctds/bge-large-en-v1.5-bio-mapping"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# quick test: encode a sample sentence
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
```

Now run the test. Note, this will locally transfer about 4GB of data and then load
that size model to do some encoding (so it may take a little time depending on your hardware).

```bash
HF_HUB_CACHE=~/.cache/gen3 HF_TOKEN=foobar HF_ENDPOINT=http://0.0.0.0:4099 python test_hf.py
```

you can also test the HF CLI, download it first. Then:

```bash
HF_HUB_CACHE=~/.cache/gen3 HF_TOKEN=foobar HF_ENDPOINT=http://0.0.0.0:4099 hf download uc-ctds/bge-large-en-v1.5-bio-mapping
```

et voilà.
