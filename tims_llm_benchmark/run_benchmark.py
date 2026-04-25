import argparse
import datetime
import json
import os
import pkg_resources
import shlex
import subprocess
import time

import requests
import yaml


parser = argparse.ArgumentParser(
    prog="python3 check_models.py",
    description="Before running check_models.py, please make sure you installed ollama successfully         on macOS, Linux, or on Windows Powershell. You can check the website: https://ollama.com",
    epilog="Author: Jason Chuang")

parser.add_argument("-v",
                    "--verbose",
                    action="store_true",
                    help="this program helps you check whether you have ollama benchmark models installed")

parser.add_argument("-m",
                    "--models",
                    type=str,
                    help="provide benchmark models YAML file path. ex. ../data/benchmark_models.yml")

parser.add_argument("-b",
                    "--benchmark",
                    type=str,
                    help="provide benchmark YAML file path. ex. ../data/benchmark1.yml")

parser.add_argument("-t",
                    "--type",
                    type=str,
                    help="provide benchmark model type. ex, instruct")

parser.add_argument("--ollamabin",
                    type=str,
                    default="ollama",
                    help="path to ollama binary")

parser.add_argument("--contextlengthbenchmark",
                    action="store_true",
                    help="run context length scaling benchmark")

parser.add_argument("--contextlengthsteps",
                    type=int,
                    default=10,
                    help="number of context length test points")


def parse_yaml(yaml_file_path):
    with open(yaml_file_path, 'r', encoding='utf-8') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            data = {}
    return data


def _model_name(entry):
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        return str(entry.get('model', '')).strip()
    return ''


def _run_unload_command(models_dict, model_name):
    unload_command = str(models_dict.get('unload_command', '')).strip()
    if not unload_command:
        return

    model_quoted = shlex.quote(model_name)
    cmd = unload_command.replace('{model}', model_quoted)
    print(f"Running unload command: {cmd}")
    subprocess.run(["bash", "-lc", cmd], check=True)

    pause_seconds = float(models_dict.get('unload_pause_seconds', 0))
    if pause_seconds > 0:
        print(f"waiting {pause_seconds:g}s before next model")
        time.sleep(pause_seconds)


def _generate_context_lengths(models_dict, steps):
    min_tokens = int(models_dict.get('context_length_min_tokens', 256))
    max_tokens = int(models_dict.get('context_length_max_tokens', 8192))
    if min_tokens < 32:
        min_tokens = 32
    if max_tokens < min_tokens:
        max_tokens = min_tokens

    if steps <= 1:
        return [min_tokens]

    ratio = (max_tokens / float(min_tokens)) ** (1.0 / float(steps - 1))
    lengths = []
    for idx in range(steps):
        value = int(round(min_tokens * (ratio ** idx)))
        if not lengths or lengths[-1] != value:
            lengths.append(value)

    if lengths[-1] != max_tokens:
        lengths[-1] = max_tokens

    return lengths


def _build_context_prompt(target_tokens, question):
    filler = "benchmarkcontext " * target_tokens
    return f"You are in a benchmark. Ignore filler context and answer only the final question.\n\nFiller:\n{filler}\n\nQuestion: {question}"


def _build_prompt_entries(models_dict, benchmark_dict, model_type, contextlengthbenchmark, contextlengthsteps):
    if contextlengthbenchmark:
        question = str(models_dict.get('context_length_question', 'What are the main causes of the American Civil War?')).strip()
        lengths = _generate_context_lengths(models_dict, contextlengthsteps)
        return [{
            'prompt': _build_context_prompt(target_tokens, question),
            'target_tokens': target_tokens,
        } for target_tokens in lengths]

    entries = []
    for one_model_type in benchmark_dict.get('modeltypes', []):
        if one_model_type.get('type') == model_type:
            for one_prompt in one_model_type.get('prompts', []):
                prompt_text = str(one_prompt.get('prompt', '')).strip()
                if prompt_text:
                    entries.append({'prompt': prompt_text, 'target_tokens': None})

    return entries


def _run_ollama_benchmark(models_dict, benchmark_dict, model_type, ollamabin, contextlengthbenchmark=False, contextlengthsteps=10):
    allowed_models = {_model_name(e) for e in models_dict.get('models', []) if _model_name(e)}
    ans = {}
    prompt_entries = _build_prompt_entries(models_dict, benchmark_dict, model_type, contextlengthbenchmark, contextlengthsteps)

    if contextlengthbenchmark:
        print(f"Running context-length benchmark ({len(prompt_entries)} steps)")

    if model_type == 'custom-model':
        print("Running custom-model")
        for model in models_dict.get('models', []):
            normalized_model = _model_name(model)
            if not normalized_model:
                continue
            for one_model_type in benchmark_dict.get('modeltypes', []):
                if one_model_type.get('type') == 'custom-model':
                    if one_model_type.get('models') is None:
                        one_model_type['models'] = []
                    one_model_type['models'].append({'model': normalized_model})

    for one_model_type in benchmark_dict.get('modeltypes', []):
        if one_model_type.get('type') != model_type:
            continue

        for onemodel in one_model_type.get('models', []):
            model_name = _model_name(onemodel)
            if model_name not in allowed_models:
                continue

            loc_dt = datetime.datetime.today()
            with open(f'log_{loc_dt.strftime("%Y-%m-%d-%H%M%S")}.log', "w", encoding='utf-8') as file1:
                stored_nums = []
                print(f'model_name =    {model_name}')
                file1.write(f'\nmodel_name =    {model_name}\n')

                try:
                    if model_name.startswith('llava'):
                        for one_prompt in one_model_type.get('prompts', []):
                            img_file_names = one_prompt.get('keywords', '').split(',')
                            for img in img_file_names:
                                img_file_path = pkg_resources.resource_filename('tims_llm_benchmark', f'data/img/{img}')
                                prompt_text = one_prompt.get('prompt', '')
                                prompt = f"{prompt_text} {img_file_path}"
                                print(f"prompt = {prompt}")
                                result = subprocess.run([ollamabin, 'run', model_name, prompt_text, '--verbose'], capture_output=True, text=True, check=True, encoding='utf-8')
                                std_err = result.stderr
                                file1.write(std_err)

                                for line in std_err.split('\n'):
                                    if ('eval rate' in line) and ('prompt' not in line):
                                        print(line)
                                        number = float(line[-20:-8])
                                        stored_nums.append(number)
                    else:
                        prompts_to_run = prompt_entries if contextlengthbenchmark else one_model_type.get('prompts', [])
                        for one_prompt in prompts_to_run:
                            if contextlengthbenchmark:
                                prompt_text = one_prompt.get('prompt', '')
                                target_tokens = one_prompt.get('target_tokens')
                            else:
                                prompt_text = one_prompt.get('prompt', '')
                                target_tokens = None

                            if target_tokens is not None:
                                print(f"context target tokens = {target_tokens}")
                                print("prompt = [context-length prompt omitted]")
                            else:
                                print(f"prompt = {prompt_text}")
                            result = subprocess.run([ollamabin, 'run', model_name, prompt_text, '--verbose'], capture_output=True, text=True, check=True, encoding='utf-8')
                            std_err = result.stderr
                            file1.write(std_err)

                            for line in std_err.split('\n'):
                                if ('eval rate' in line) and ('prompt' not in line):
                                        if target_tokens is not None:
                                            print(f"{line} (target context {target_tokens})")
                                        else:
                                            print(line)
                                        number = float(line[-20:-8])
                                        stored_nums.append(number)
                finally:
                    _run_unload_command(models_dict, model_name)

                print("-" * 20)
                if stored_nums:
                    average = sum(stored_nums) / len(stored_nums)
                    print("Average of eval rate: ", round(average, 3), " tokens/s")
                    ans[f"{model_name}"] = f"{round(average, 3):.2f}"

                print("-" * 40)
                file1.write("\n" + "-" * 40)

    return ans


def _stream_openai_response(response):
    completion_tokens = None
    prompt_tokens = None
    generated_text = []
    first_token_ts = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith('data:'):
            continue

        line = raw_line[5:].strip()
        if line == '[DONE]':
            break

        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        usage = chunk.get('usage')
        if isinstance(usage, dict) and usage.get('completion_tokens') is not None:
            completion_tokens = int(usage['completion_tokens'])
            if usage.get('prompt_tokens') is not None:
                prompt_tokens = int(usage['prompt_tokens'])
            if first_token_ts is None:
                first_token_ts = time.perf_counter()

        choices = chunk.get('choices', [])
        if choices:
            if first_token_ts is None:
                first_token_ts = time.perf_counter()
            delta = choices[0].get('delta', {})
            content = delta.get('content')
            if content:
                generated_text.append(content)

    return completion_tokens, prompt_tokens, ''.join(generated_text), first_token_ts


def _run_openai_benchmark(models_dict, benchmark_dict, model_type, contextlengthbenchmark=False, contextlengthsteps=10):
    endpoint = str(models_dict.get('endpoint', '')).strip()
    if not endpoint:
        raise ValueError('OpenAI-compatible benchmark requires endpoint in models yaml')

    api_key_env = str(models_dict.get('api_key_env', 'OPENAI_API_KEY')).strip() or 'OPENAI_API_KEY'
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f'Missing required API key environment variable: {api_key_env}')

    request_defaults = models_dict.get('request_defaults', {}) or {}
    if not isinstance(request_defaults, dict):
        raise ValueError('request_defaults must be a map')

    host_header = str(models_dict.get('host_header', '')).strip()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    if host_header:
        headers['Host'] = host_header

    prompt_entries = _build_prompt_entries(models_dict, benchmark_dict, model_type, contextlengthbenchmark, contextlengthsteps)
    prompts = [entry['prompt'] for entry in prompt_entries]

    if not prompts:
        raise ValueError(f'No prompts found for benchmark type: {model_type}')

    print('Running custom-model')
    if contextlengthbenchmark:
        print(f'context-length mode: {len(prompts)} steps')
    print(f'endpoint: {endpoint}')
    if host_header:
        print(f'host_header: {host_header}')

    ans = {}
    for model_entry in models_dict.get('models', []):
        model_name = _model_name(model_entry)
        if not model_name:
            continue

        loc_dt = datetime.datetime.today()
        with open(f'log_{loc_dt.strftime("%Y-%m-%d-%H%M%S")}.log', "w", encoding='utf-8') as file1:
            stored_nums = []
            ttft_ms_values = []
            context_points = []
            print(f'model_name =    {model_name}')
            file1.write(f'\nmodel_name =    {model_name}\n')

            try:
                for idx, prompt in enumerate(prompts):
                    target_context = prompt_entries[idx].get('target_tokens')
                    if target_context is not None:
                        print(f'context target tokens = {target_context}')
                        print('prompt = [context-length prompt omitted]')
                    else:
                        print(f'prompt = {prompt}')
                    payload = {
                        'model': model_name,
                        'messages': [{'role': 'user', 'content': prompt}],
                    }
                    payload.update(request_defaults)

                    if 'stream' not in payload:
                        payload['stream'] = True
                    if payload.get('stream') and 'stream_options' not in payload:
                        payload['stream_options'] = {'include_usage': True}

                    req_start = time.perf_counter()
                    with requests.post(endpoint, headers=headers, json=payload, stream=bool(payload.get('stream')), timeout=(20, 1200)) as response:
                        response.raise_for_status()

                        if payload.get('stream'):
                            completion_tokens, prompt_tokens, generated_text, first_token_ts = _stream_openai_response(response)
                        else:
                            body = response.json()
                            usage = body.get('usage', {})
                            completion_tokens = usage.get('completion_tokens')
                            prompt_tokens = usage.get('prompt_tokens')
                            if completion_tokens is not None:
                                completion_tokens = int(completion_tokens)
                            if prompt_tokens is not None:
                                prompt_tokens = int(prompt_tokens)
                            generated_text = body.get('choices', [{}])[0].get('message', {}).get('content', '')
                            first_token_ts = req_start

                    req_end = time.perf_counter()
                    if first_token_ts is None:
                        first_token_ts = req_start

                    ttft_seconds = max(first_token_ts - req_start, 0.0)
                    ttft_ms = ttft_seconds * 1000.0
                    ttft_ms_values.append(ttft_ms)

                    decode_seconds = max(req_end - first_token_ts, 1e-9)
                    if completion_tokens is None:
                        completion_tokens = len(generated_text.split())

                    eval_rate = float(completion_tokens) / decode_seconds
                    stored_nums.append(eval_rate)
                    print(f'eval rate:            {eval_rate:0.2f} tokens/s')
                    print(f'ttft:                 {ttft_ms:0.2f} ms')
                    if prompt_tokens is not None:
                        print(f'prompt tokens:        {prompt_tokens}')
                    if target_context is not None:
                        file1.write('prompt = [context-length prompt omitted]\n')
                    else:
                        file1.write(f'prompt = {prompt}\n')
                    file1.write(f'eval rate:            {eval_rate:0.2f} tokens/s\n')
                    file1.write(f'ttft:                 {ttft_ms:0.2f} ms\n')
                    if prompt_tokens is not None:
                        file1.write(f'prompt tokens:        {prompt_tokens}\n')

                    if target_context is not None:
                        context_points.append({
                            'target_context_tokens': target_context,
                            'prompt_tokens': prompt_tokens,
                            'eval_rate': eval_rate,
                            'ttft_ms': ttft_ms,
                        })
            finally:
                _run_unload_command(models_dict, model_name)

            print('-' * 20)
            if stored_nums:
                average = sum(stored_nums) / len(stored_nums)
                print('Average of eval rate: ', round(average, 3), ' tokens/s')
                ans[f'{model_name}'] = f'{round(average, 3):.2f}'
            if ttft_ms_values:
                avg_ttft = sum(ttft_ms_values) / len(ttft_ms_values)
                print('Average TTFT: ', round(avg_ttft, 2), ' ms')
                file1.write(f'\nAverage TTFT: {avg_ttft:0.2f} ms\n')

            if context_points:
                print('Context-length results:')
                file1.write('\nContext-length results:\n')
                for point in context_points:
                    prompt_tokens_text = point['prompt_tokens'] if point['prompt_tokens'] is not None else 'n/a'
                    line = f"target={point['target_context_tokens']}, prompt_tokens={prompt_tokens_text}, eval_rate={point['eval_rate']:.2f} tokens/s, ttft={point['ttft_ms']:.2f} ms"
                    print(line)
                    file1.write(line + '\n')
            print('-' * 40)
            file1.write('\n' + '-' * 40)

    return ans


def run_benchmark(models_file_path, benchmark_file_path, type, ollamabin: str = 'ollama', apibenchmark: bool = False, contextlengthbenchmark: bool = False, contextlengthsteps: int = 10):
    models_dict = parse_yaml(models_file_path)
    benchmark_dict = parse_yaml(benchmark_file_path)
    provider = str(models_dict.get('provider', 'ollama')).strip().lower()

    if contextlengthsteps < 1:
        raise ValueError('contextlengthsteps must be >= 1')

    if apibenchmark or provider in ('openai-compatible', 'litellm'):
        return _run_openai_benchmark(models_dict, benchmark_dict, type, contextlengthbenchmark, contextlengthsteps)

    return _run_ollama_benchmark(models_dict, benchmark_dict, type, ollamabin, contextlengthbenchmark, contextlengthsteps)


if __name__ == "__main__":
    args = parser.parse_args()
    if (args.models is not None) and (args.benchmark is not None) and (args.type is not None):
        run_benchmark(args.models, args.benchmark, args.type, args.ollamabin, False, args.contextlengthbenchmark, args.contextlengthsteps)
        print('-' * 40)
