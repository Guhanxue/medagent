# Local vision-language model

The default release configuration uses the open-weight [Qwen3-VL 4B Instruct model](https://ollama.com/library/qwen3-vl%3A4b-instruct) through [Ollama’s OpenAI-compatible endpoint](https://docs.ollama.com/api/openai-compatibility). This supports text and image messages. The main paper benchmarks use GPT-5.2. A [focused comparison](PLANNERS.md) evaluates Qwen-4B and Qwen-8B as action planners using official bfloat16 weights, while keeping GPT-5.2 image analysis and critique fixed. It also includes GPT-5.6 Luna through a subscription-based Codex CLI configuration.

## Start and check

Install Ollama using its official installer. Start its application or run `ollama serve` in a separate terminal. Do not start a second server if one is already running.

```sh
ollama pull qwen3-vl:4b-instruct
export MEDADAPT_LLM_BACKEND=local
export MEDADAPT_LLM_BASE_URL=http://127.0.0.1:11434/v1
export MEDADAPT_LLM_MODEL=qwen3-vl:4b-instruct
medadapt --check-llm
```

The first command downloads model weights. Memory and runtime depend on the selected quantization, image size and context. The final command sends only a generated red image. Success confirms a working text/image request; it does not validate medical reasoning or adaptation performance.

`.env.example` lists the settings; the program does not automatically read `.env`. Export the variables in the shell used to run `medadapt`.

## Another compatible server

For an existing vLLM or other OpenAI-compatible vision endpoint, set `MEDADAPT_LLM_BASE_URL` and `MEDADAPT_LLM_MODEL` to that deployment's URL and model identifier. Use `MEDADAPT_LLM_API_KEY` only if that endpoint requires authentication. Remote endpoints require HTTPS; unencrypted HTTP is accepted only on loopback. Localhost traffic bypasses environment proxies. The client does not switch providers or models after a failure.

The `local` backend sends standard `messages` with text/image content, `max_tokens`, `temperature`, and `seed`. The server must support those parameters and return a completed chat response. Models that exceed the output limit or return an invalid tool action cause an explicit failure/escalation. Inspect `run.json` and server configuration before retrying.

For an explicitly selected cloud backend, use `--backend openai --model YOUR_MODEL` with `OPENAI_API_KEY`, or `--backend azure --model YOUR_DEPLOYMENT` with `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` and `AZURE_API_VERSION`. No institutional endpoint or key is supplied. Image critique uses the OpenAI-compatible/Azure interface; the optional Anthropic interface supports legacy text calls only.

The default endpoint binds to the local machine. Publicly available model weights do not require exposing your inference server publicly. No hosted service is provisioned by this repository.
