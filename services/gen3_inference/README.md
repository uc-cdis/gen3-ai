# Gen3 Inference


API is compliant with Open Responses.

## Open Responses Conformance Testing

Clone the [openresponses repo](https://github.com/openresponses/openresponses).

Change directory into where you cloned.

Ensure Gen3 Inference is running locally. `just run gen3_inferece`

You need bun for openresponses, see https://bun.com/docs/installation.

Ensure skip auth is enabled.

Run:

```bash
bun run test:compliance --base-url http://localhost:4143 --api-key "foobar"
```
