from gen3_ai_model_repo.models.schemas import (
    RepositoryMetadataModel,
    RevisionModel,
    TreeEntryModel,
)


def repository_metadata_to_model(data: dict) -> RepositoryMetadataModel:
    """
    Convert repository metadata dict into schema model.
    """
    return RepositoryMetadataModel(
        namespace=data["namespace"],
        repo=data.get("repo") or data.get("repo_name"),
        description=data["description"],
        tags=data.get("tags", []),
        created_at=data.get("created_at", ""),
    )


def revision_to_model(revision_data: dict) -> RevisionModel:
    """
    Convert revision data into API schema.
    """
    return RevisionModel(
        id=revision_data["id"],
        revision=revision_data["revision"],
        sha=revision_data["sha"],
    )


def repository_file_to_model(file_data: dict) -> TreeEntryModel:
    """
    Convert repository file metadata into API schema.
    """
    return TreeEntryModel(
        type=file_data["type"],
        oid=file_data["oid"],
        size=file_data["size"],
    )
