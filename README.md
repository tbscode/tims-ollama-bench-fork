# llm-benchmark (Local Fork)

*This is a privacy-focused fork of [aidatatools/ollama-benchmark](https://github.com/aidatatools/ollama-benchmark) with all upstream telemetry and result reporting removed.*

Measure how fast your local LLMs *really* are with a simple, cross-platform CLI tool that tells you the tokens-per-second truth via Ollama.

### Prerequisites

- A working [Ollama](https://ollama.com) installation.

### Installation

You can install this package directly from the github repository using pip:

```bash
pip install git+https://github.com/tbscode/tims-ollama-bench-fork.git
```

### Usage

Run the standard benchmark:

```bash
llm_benchmark run
```

*Note: The benchmark will automatically detect your system's RAM and pull appropriate models via Ollama before running.*

### Advanced Usage

#### Custom Ollama Path
If you built your own developer version of Ollama:
```bash
llm_benchmark run --ollamabin=~/code/ollama/ollama
```

#### Custom Benchmark Models
You can provide a custom YAML file to benchmark specific models:
```yaml
# custombenchmarkmodels.yml
file_name: "custombenchmarkmodels.yml"
version: 2.0.custom
models:
  - model: "deepseek-r1:1.5b"
  - model: "qwen:0.5b"
```

Run with the custom flag:
```bash
llm_benchmark run --custombenchmark=path/to/custombenchmarkmodels.yml
```