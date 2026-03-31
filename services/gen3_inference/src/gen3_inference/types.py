import json
from typing import Annotated

from openresponses_types import Error
from pydantic import Field


class OpenResponsesError(Error):
    """
    The written Open Responses spec as of now says type is required and code is
    option, but the openAPI spec is the opposite? So we'll just include both
    """

    type: Annotated[
        str,
        Field(
            description="The category of the error, such as server_error, model_error, invalid_request, or not_found. These generally, but not always, map to the status code of the response."
        ),
    ]

    def to_json(self) -> dict:
        return {"error": json.loads(self.model_dump_json())}
