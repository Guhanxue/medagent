# Credential and artifact handling

The default local backend uses no author, institutional, or paid-provider API key. Configuration comes from environment variables. `.env` files are not automatically loaded. Azure/OpenAI credentials are read only when that backend is explicitly selected; the local backend uses only `MEDADAPT_LLM_API_KEY` if supplied.

The release provider client rejects credentials in endpoint URLs, remote plain HTTP, and redirects. Request errors omit server response bodies and credentials. Keep provider SDK debug logging disabled when handling private inputs.

`.gitignore` excludes local environment files, private-key files, data, checkpoints and runtime outputs. Ignore rules do not remove previously committed content. Before sharing a revision, inspect its staged files and scan the working tree and full history with [Gitleaks](https://github.com/gitleaks/gitleaks):

```sh
gitleaks dir --redact=100 .
gitleaks git --redact=100 --log-opts="--all" .
```

Also inspect PDFs and archives; scanners cannot guarantee detection of every secret format. Never upload actual keys or private clinical data to tests, issues, logs or examples.

Load only trusted checkpoints and generated inference packages. Serialized preprocessing functions can execute Python code and may contain captured reference-image information. They belong in local ignored output directories, not in the source repository. Cached LLM responses are opt-in and should also remain private.

If a credential is exposed, revoke or rotate it with its issuer, then remove it from repository history and distributed artifacts. Report the affected path privately without copying the credential value.
