import base64
import numpy
import requests
import os

response = requests.post(
    "http://127.0.0.1:4142/vectorstore/collections/summ/embeddings/bulk?no_embeddings_info=true",
    headers={"Authorization": f"Bearer {os.environ['GEN3_ACCESS_TOKEN']}"},
    json=[
        # expr, vector (float32)
        # "1e7e1ac9-b299-4382-8fcd-a64055d6d8b6",
        # "8995539d-13bc-41c2-80df-6ff41021749f",
        # summ, halfvect (float16)
        "5b90d2c6-eb11-4a78-a6a6-5f7907ce8a6c",
        "735c5fba-fe39-4ed2-a27a-0df3afc8568d",
    ],
)
response.raise_for_status()

precision = response.json()["precision"]
vector_dtype = numpy.float16 if precision == "float16" else numpy.float32

all_embeddings = []
for embedding in response.json()["embeddings"]:
    b64_str = embedding["vector_base64"]

    # re-pad the string to a multiple of 4 (handles any missing '=' signs)
    padding_needed = -len(b64_str) % 4
    padded_b64 = b64_str + ("=" * padding_needed)

    decoded_bytes = base64.urlsafe_b64decode(padded_b64)

    decoded_array = numpy.frombuffer(decoded_bytes, dtype=vector_dtype)

    print("Decoded Array:", decoded_array[:5], "...")
    print("Data Type:", decoded_array.dtype)
    all_embeddings.append(decoded_array)

# verification
print(f"Successfully parsed {len(all_embeddings)} vectors")
