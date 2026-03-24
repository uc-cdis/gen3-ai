# VSCode

## Debugging

This comes bundled with a `launch.json` which uses uvicorn to launch the apps.

This allows you to easily debug in VSCode.

Ensure you have installed the deps with `just install`.

Ensure you have the Python interpreter in VS Code set to the service `.venv/bin/python` for the service you want to debug.

Go to the "Run and Debug" pane and you should see each of the services available to run:

- `gen3_inference`
- `gen3_embeddings`
- `gen3_ai_model_repo`

## Importing Common Libraries

You may need to add this to your .vscode/settings.json for intellisense to pick up the common libraries:

```
"python.analysis.extraPaths": [
    "./libraries/common",
    "./libraries/common/src"
]
```
