"""
HF example of inference provider mapping:

https://github.com/huggingface/huggingface_hub/blob/176bdfb9c1459f5c5f0b70a2ca6b2b9fa02dffc1/src/huggingface_hub/hf_api.py#L796-L819

inference_provider_mapping: [
    {
        "provider_name": "openresponses",
        ""
    }
]

Model Info:

https://github.com/huggingface/huggingface_hub/blob/176bdfb9c1459f5c5f0b70a2ca6b2b9fa02dffc1/src/huggingface_hub/hf_api.py#L822

"""

from abc import ABC, abstractmethod

from openresponses_types import CreateResponseBody


class InferenceProtocolClient(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def generate_non_streaming_response(self, body: CreateResponseBody):
        raise NotImplementedError()

    @abstractmethod
    async def generate_streaming_response(self, body: CreateResponseBody):
        raise NotImplementedError()
