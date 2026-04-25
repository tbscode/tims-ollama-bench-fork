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


def _run_ollama_benchmark(models_dict, benchmark_dict, model_type, ollamabin):
    allowed_models = {_model_name(e) for e in models_dict.get('models', []) if _model_name(e)}
    ans = {}

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
                        for one_prompt in one_model_type.get('prompts', []):
                            prompt_text = one_prompt.get('prompt', '')
                            print(f"prompt = {prompt_text}")
                            result = subprocess.run([ollamabin, 'run', model_name, prompt_text, '--verbose'], capture_output=True, text=True, check=True, encoding='utf-8')
                            std_err = result.stderr
                            file1.write(std_err)

                            for line in std_err.split('\n'):
                                if ('eval rate' in line) and ('prompt' not in line):
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

        choices = chunk.get('choices', [])
        if choices:
            delta = choices[0].get('delta', {})
            content = delta.get('content')
            if content:
                generated_text.append(content)
                if first_token_ts is None:
                    first_token_ts = time.perf_counter()

    return completion_tokens, ''.join(generated_text), first_token_ts


def _run_openai_benchmark(models_dict, benchmark_dict, model_type):
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

    prompts = []
    for one_model_type in benchmark_dict.get('modeltypes', []):
        if one_model_type.get('type') == model_type:
            for one_prompt in one_model_type.get('prompts', []):
                prompt_text = str(one_prompt.get('prompt', '')).strip()
                if prompt_text:
                    prompts.append(prompt_text)

    if not prompts:
        raise ValueError(f'No prompts found for benchmark type: {model_type}')

    print('Running custom-model')
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
            print(f'model_name =    {model_name}')
            file1.write(f'\nmodel_name =    {model_name}\n')

            try:
                for prompt in prompts:
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
                            completion_tokens, generated_text, first_token_ts = _stream_openai_response(response)
                        else:
                            body = response.json()
                            usage = body.get('usage', {})
                            completion_tokens = usage.get('completion_tokens')
                            if completion_tokens is not None:
                                completion_tokens = int(completion_tokens)
                            generated_text = body.get('choices', [{}])[0].get('message', {}).get('content', '')
                            first_token_ts = req_start

                    req_end = time.perf_counter()
                    if first_token_ts is None:
                        first_token_ts = req_start

                    decode_seconds = max(req_end - first_token_ts, 1e-9)
                    if completion_tokens is None:
                        completion_tokens = len(generated_text.split())

                    eval_rate = float(completion_tokens) / decode_seconds
                    stored_nums.append(eval_rate)
                    print(f'eval rate:            {eval_rate:0.2f} tokens/s')
                    file1.write(f'prompt = {prompt}\n')
                    file1.write(f'eval rate:            {eval_rate:0.2f} tokens/s\n')
            finally:
                _run_unload_command(models_dict, model_name)

            print('-' * 20)
            if stored_nums:
                average = sum(stored_nums) / len(stored_nums)
                print('Average of eval rate: ', round(average, 3), ' tokens/s')
                ans[f'{model_name}'] = f'{round(average, 3):.2f}'
            print('-' * 40)
            file1.write('\n' + '-' * 40)

    return ans


def run_benchmark(models_file_path, benchmark_file_path, type, ollamabin: str = 'ollama', apibenchmark: bool = False):
    models_dict = parse_yaml(models_file_path)
    benchmark_dict = parse_yaml(benchmark_file_path)
    provider = str(models_dict.get('provider', 'ollama')).strip().lower()

    if apibenchmark or provider in ('openai-compatible', 'litellm'):
        return _run_openai_benchmark(models_dict, benchmark_dict, type)

    return _run_ollama_benchmark(models_dict, benchmark_dict, type, ollamabin)


if __name__ == "__main__":
    args = parser.parse_args()
    if (args.models is not None) and (args.benchmark is not None) and (args.type is not None):
        run_benchmark(args.models, args.benchmark, args.type, args.ollamabin, False)
        print('-' * 40)
