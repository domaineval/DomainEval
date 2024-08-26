import os
import json
import argparse
from typing import List, Dict
from tqdm import tqdm
from utils import check_correctness
import concurrent.futures
from multiprocessing import Manager

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

def modify_code(code_response : str, code_reference : str) -> str:
    # Package dependency supplement, after filtering through sandbox. py, has been rewritten as one line at a time without cross line import statements
    for line in code_reference.split('\n'):
        if line.startswith("import ") or line.startswith("from "):
            if line not in code_response:
                code_response = line + "\n" + code_response
    # Move all import statements from __future__ to the beginning
    future_import_lines = [line for line in code_response.split('\n') if 'from __future__ import ' in line]
    code_without_future = [line for line in code_response.split('\n') if 'from __future__ import ' not in line]
    code_response = "\n".join(future_import_lines + code_without_future).lstrip()
    
    return code_response

def exec_response_datapoint(data: Dict, debug : bool = False) -> Dict:
    if not data["response"]: result = "No code found"
    else:
        for test_dict in data["test_code_list"]:
            exec_code = test_dict["code_start"] + "\n" + data["response"] + "\n" + test_dict["test_code"]
            exec_code = modify_code(code_response = exec_code, code_reference = data["method_code_mask"])
            # Relaxation of time_out restriction during actual testing
            exec_result = check_correctness(exec_code, timeout = 20.0)
            result = exec_result['result']
            if exec_result["passed"] != True: break
    if debug:
        print(f"\nexec_code:\n\n{exec_code}\n")
        print(f"\nresult: {result}\n")
    exec_result_dict = {
        "result": result,
        "model_name": data["model_name"],
        "repository": data["repository"],
        "full_method_name": data["full_method_name"],
        "method_path": data["method_path"],
        "method_code_mask": data["method_code_mask"],
        "response": data["response"],
    }
    return exec_result_dict

def task(data : Dict, result_list):
    result = exec_response_datapoint(data = data)
    result_list.append(result)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Eval_Generate")
    parser.add_argument("--model_name", "-m", type=str, default="std", help="Model Name")
    parser.add_argument("--version", "-v", type=str, default="domaineval_20240711", help="Version")
    parser.add_argument("--generate_root_dir", "-g", type=str, default="modelresult", help="Eval_Generate Directory")
    parser.add_argument("--k_pass", "-k", type=int, default=1, help="pass@k")
    parser.add_argument("--output_root_dir", "-o", type=str, default="executeresult", help="Result Directory")
    args = parser.parse_args()
    
    generate_dir = os.path.join(args.generate_root_dir, args.version, args.model_name, f"pass_{args.k_pass}")
    if not os.path.exists(generate_dir):
        raise FileNotFoundError(f"Result Directory {generate_dir} Not Found")
    if args.k_pass < 1: raise ValueError("k in pass@k must be greater than 0")
    
    print(f"Eval_Generate Directory: {generate_dir}")
    print(f"Model: {args.model_name}")
    
    for topic in os.listdir(generate_dir):
        topic_path = os.path.join(generate_dir, topic)
        if not os.path.isdir(topic_path): continue
        
        print(f"\nExecute topic: {topic}")
        for file in os.listdir(topic_path):
            if not file.endswith(".jsonl"): continue
            
            exec_path = topic_path.replace(args.generate_root_dir, args.output_root_dir)
            if not os.path.exists(exec_path): os.makedirs(exec_path)
            
            exec_path = os.path.join(exec_path, file)
            if os.path.exists(exec_path):
                print(f"{exec_path} already exists")
                continue
            
            generate_jsonl_path = os.path.join(topic_path, file)
            with open(generate_jsonl_path, "r", encoding = 'utf-8') as f:
                lines = f.readlines()
            data_list = [json.loads(line) for line in lines]
            if len(data_list) <= 0: continue
            print(f"Execute {topic}/ {file}")
            print(f"len(data_list) = {len(data_list)}")
            
            manager = Manager()
            result_list = manager.list()
            assert len(result_list) == 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(task, data, result_list) for data in data_list]
                concurrent.futures.wait(futures)
            exec_result_list = list(result_list)
            
            if len(exec_result_list) <= 0: continue
            with open(exec_path, "w", encoding = 'utf-8') as f:
                for result in exec_result_list:
                    f.write(json.dumps(result) + "\n")
            
            print(f"Execute {topic}/ {file} End")
        print(f"\nExecute topic: {topic} End")
    print(f"Execute End")