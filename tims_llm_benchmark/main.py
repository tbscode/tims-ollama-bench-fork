import base64
import argparse
import sys
from tims_llm_benchmark import check_models
from tims_llm_benchmark import check_ollama
from tims_llm_benchmark import run_benchmark

from .systeminfo import sysmain

import pkg_resources

from typing import Optional

def hello(name: str):
    print(f"Hello {name}!")

def run(ollamabin: str = 'ollama', custombenchmark: Optional[str] = None):
    sys_info = sysmain.get_extra()
    print(f"Total memory size : {sys_info['memory']:.2f} GB") 
    print(f"cpu_info: {sys_info['cpu']}")
    print(f"gpu_info: {sys_info['gpu']}")
    print(f"os_version: {sys_info['os_version']}")

    ollama_version = check_ollama.check_ollama_version(ollamabin)
    print(f"ollama_version: {ollama_version}")
    print('-'*10)

    ft_mem_size = float(f"{sys_info['memory']:.2f}")

    if custombenchmark:
        models_file_path = custombenchmark
        print(f"running custom benchmark from models_file_path: {models_file_path}")
    else:
        models_file_path = pkg_resources.resource_filename('tims_llm_benchmark','data/benchmark_models_32gb_ram.yml')
        if(ft_mem_size>=1 and ft_mem_size <2):
            models_file_path = pkg_resources.resource_filename('tims_llm_benchmark','data/benchmark_models_2gb_ram.yml')
        elif(ft_mem_size>=2 and ft_mem_size <4):
            models_file_path = pkg_resources.resource_filename('tims_llm_benchmark','data/benchmark_models_3gb_ram.yml')
        elif(ft_mem_size>=4 and ft_mem_size <7):
            models_file_path = pkg_resources.resource_filename('tims_llm_benchmark','data/benchmark_models_4gb_ram.yml')
        elif(ft_mem_size>=7 and ft_mem_size <15):
            models_file_path = pkg_resources.resource_filename('tims_llm_benchmark','data/benchmark_models_8gb_ram.yml')
        elif(ft_mem_size>=15 and ft_mem_size <31):
            models_file_path = pkg_resources.resource_filename('tims_llm_benchmark','data/benchmark_models_16gb_ram.yml')

    check_models.pull_models(models_file_path)
    print('-'*10)

    benchmark_file_path = pkg_resources.resource_filename('tims_llm_benchmark','data/benchmark2.yml')

    bench_results_info = {}
    is_simulation = False
    if custombenchmark:
        result0 = run_benchmark.run_benchmark(models_file_path,benchmark_file_path, 'custom-model', ollamabin)
        bench_results_info.update(result0)
    elif is_simulation==False :
        result1 = run_benchmark.run_benchmark(models_file_path,benchmark_file_path, 'instruct', ollamabin)
        bench_results_info.update(result1)
        result2 = run_benchmark.run_benchmark(models_file_path,benchmark_file_path, 'question-answer', ollamabin)
        bench_results_info.update(result2)
        result3 = run_benchmark.run_benchmark(models_file_path,benchmark_file_path, 'vision-image', ollamabin)
        bench_results_info.update(result3)
        result4 = run_benchmark.run_benchmark(models_file_path,benchmark_file_path, 'instruction-question-answer-code-generation', ollamabin)
        bench_results_info.update(result4)
    else:
        bench_results_info.update({"llama2:7b":7.65})
        bench_results_info.update({"gemma2:7b":17.77})

def goodbye(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Mr.(Ms.) {name}. Have a good day.")
    else:
        print(f"Bye {name}!")

def sysinfo(formal: bool = True):
    if formal:
        sys_info = sysmain.get_extra()
        print(f"memory : {sys_info['memory']:.2f} GB") 
        print(f"cpu_info: {sys_info['cpu']}")
        print(f"gpu_info: {sys_info['gpu']}")
        print(f"os_version: {sys_info['os_version']}")
    else:
        print(f"No print!")

def app():
    parser = argparse.ArgumentParser(description="LLM Benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the benchmark")
    run_parser.add_argument("--ollamabin", type=str, default="ollama", help="Path to ollama binary")
    run_parser.add_argument("--custombenchmark", type=str, default=None, help="Path to custom benchmark models yaml")

    # sysinfo command
    sysinfo_parser = subparsers.add_parser("sysinfo", help="Print system information")
    sysinfo_parser.add_argument("--formal", action=argparse.BooleanOptionalAction, default=True, help="Formal print output")

    # hello command
    hello_parser = subparsers.add_parser("hello", help="Say hello")
    hello_parser.add_argument("name", type=str, help="Name to greet")

    # goodbye command
    goodbye_parser = subparsers.add_parser("goodbye", help="Say goodbye")
    goodbye_parser.add_argument("name", type=str, help="Name to say goodbye to")
    goodbye_parser.add_argument("--formal", action="store_true", help="Formal goodbye")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "run":
        run(args.ollamabin, args.custombenchmark)
    elif args.command == "sysinfo":
        sysinfo(args.formal)
    elif args.command == "hello":
        hello(args.name)
    elif args.command == "goodbye":
        goodbye(args.name, args.formal)

if __name__ == "__main__":
    app()