# Gen3 Inference

API is compliant with Open Responses.

> WARNING: WORK IN PROGRESS. Service is not fully implemented yet

## Open Responses Conformance Testing

Clone the [openresponses repo](https://github.com/openresponses/openresponses).

Change directory into where you cloned.

Ensure Gen3 Inference is running locally. `just run gen3_inferece`

You need bun for openresponses, see https://bun.com/docs/installation.

Ensure skip auth is enabled.

Run:

```bash
bun run test:compliance --base-url http://localhost:4143 --api-key "foobar" -v --model llama3.2:latest
```

## TODO

- [ ] All TODO's and FIXME's in code
- [ ] implement authz
- [ ] /docs errors
- [ ] official openresponses conformance testing (function_call tests failing, streaming response failing missing verbosity)
- [ ] socket hangup on request after `gen3 run gen3_inference` but running through VSCode (uvicorn only) seems to work fine
- [ ] ensure tool requests are appropriately handled
- [ ] more comprehensive unit testing
- [ ] actually test against KServe inference services and make necessary code adjustments
- [ ] unit tests ensure MOCK_AI_MODEL_REPO_REPONSE is set to False for testing
