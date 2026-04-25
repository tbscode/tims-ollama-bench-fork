# llm-benchmark (Local Fork)

> *tracking disabled fork of [aidatatools/ollama-benchmark](https://github.com/aidatatools/ollama-benchmark) with telemetry removed.*

Used by [`@tbscode`](https://github.com/tbscode) for benchmarks published [in this blog post](https://blog.t1m.me/blog/local-llms-on-strix-halo-128gb-shared-ram).

Measure your local LLMs' tokens-per-second performance via [Ollama](https://ollama.com) or an OpenAI-compatible API router.

### Installation

```bash
pip install git+https://github.com/tbscode/tims-ollama-bench-fork.git
```

### Usage (Custom Benchmark)

The primary way to use this tool is by providing a custom YAML file specifying the models you want to test.

1. Create a `models.yml` file:
```yaml
file_name: "models.yml"
version: 2.0.custom
models:
  - model: "deepseek-r1:1.5b"
  - model: "qwen:0.5b"
```

2. Run the benchmark:
```bash
llm_benchmark run --custombenchmark=models.yml
```

*(Optional)* Specify a custom Ollama binary path:
```bash
llm_benchmark run --ollamabin=/path/to/ollama --custombenchmark=models.yml
```

### Usage (OpenAI-compatible API mode)

Enable API mode with `--apibenchmark`:

```bash
tims_llm_benchmark run --custombenchmark=models.yml --apibenchmark
```

Example `models.yml`:

```yaml
file_name: "models.yml"
version: 2.1.custom
provider: openai-compatible
endpoint: "http://tim-bosgame2/v1/chat/completions"
host_header: "litellm.home.arpa"
api_key_env: "LITELLM_KEY"
request_defaults:
  max_tokens: 120
  temperature: 0.2
  stream: true
models:
  - model: "gemma4-31b"
unload_command: "./scripts/unload-model.sh {model}"
unload_pause_seconds: 10
```
