import os
import json
import argparse
from typing import List

def collect_method(full_method_name : str, dir_path : str) -> List:
    info = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.jsonl'): 
                file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    if data["full_method_name"] == full_method_name:
                        info.append(data)
    return info

def show_dict(data : dict, keys : List = None):
    print("-"*20)
    for key in data:
        if keys is not None and key not in keys: continue
        if type(data[key]) == str:
            if "\n" in data[key]: print(f"[{key}]:\n{data[key]}")
            else: print(f"[{key}]: {data[key]}")
        if type(data[key]) == list:
            for i in range(len(data[key])):
                print(f"[{key}][{i}]:\n{data[key][i]}")
        if type(data[key]) == int:
            print(f"[{key}]: {data[key]}")
            
def same_case(data1 : dict, data2 : dict) -> bool:
    for key in data1:
        if key in data2:
            if data1[key] != data2[key]:
                return False
    return True

def case_study(
        args,
        full_method_name : str = None,
    ):
    if full_method_name is None: return
    domaineval_path = args.domaineval_path
    execresult_path = args.execresult_path
    if not os.path.exists(execresult_path): raise FileNotFoundError(f"{execresult_path} not found")
    if not os.path.exists(domaineval_path): raise FileNotFoundError(f"{domaineval_path} not found")
    
    method_info = collect_method(full_method_name, domaineval_path)
    response_info = collect_method(full_method_name, execresult_path)
    for m_data in method_info:
        show_dict(m_data, ["full_method_name", "method_path", "instruction", "method_code_mask", "test_code_list"])
        for r_data in response_info:
            if same_case(m_data, r_data) and r_data["model_name"] == "std":
                show_dict(r_data, ["result", "model_name", "response"])
        for r_data in response_info:
            if same_case(m_data, r_data) and r_data["model_name"] != "std":
                show_dict(r_data, ["result", "model_name", "response"])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domaineval_path', '-de', type=str, default="../domaineval_20240711")
    parser.add_argument('--execresult_path', '-er', type=str, default="../executeresult/domaineval_20240711")
    parser.add_argument('--full_method_name', '-fmn', type=str, default=None)

    args = parser.parse_args()
    case_study(
        full_method_name = args.full_method_name,
        args = args
    )