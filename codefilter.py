import json
import os
import ast
import astor
import shutil
import argparse
from typing import Tuple

def calc_fun_body_line(code : str, func_name : str, debug = False) -> Tuple[int, str]:
    """
    Calculate the number of lines in the body of a specified function, excluding docstrings and comments.

    Parameters:
    - code (str): The source code of the Python module as a string.
    - func_name (str): The name of the function whose body lines are to be counted.
    - debug (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
    - Tuple[int, str]: A tuple containing:
        - int: The number of lines in the function body, excluding docstrings and comments.
        - str: The source code of the function body as a string.
    """
    tree = ast.parse(code)
    body_linecount = 0
    body_linecode = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == func_name:
                body_linecode = astor.to_source(node).strip()
                for body_node in node.body:
                    # Remove docstrings and comments.
                    if not isinstance(body_node, ast.Expr) or \
                       not isinstance(body_node.value, ast.Constant) or \
                       not isinstance(body_node.value.value, str):
                        body_code = astor.to_source(body_node).strip()
                        if len(body_code) <= 0: continue
                        # body_linecode += body_code + "\n"
                        body_linecount += len(body_code.splitlines())
                        if debug: print((body_code, body_linecount))
    if body_linecount <= 0: print(f"{func_name} not found !")
    return body_linecount, body_linecode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench_dir', type=str, default="bench_20240711")
    args = parser.parse_args()
    
    bench_path = os.path.join(args.bench_dir)
    target_path = os.path.join("filter")
    
    if os.path.exists(target_path):
        try:
            shutil.rmtree(target_path)
            print(f"Deleted {target_path} successfully")
        except Exception as e:
            print(f"Error: {e}")
    
    max_bodyline = 0
    max_totalline = 0
    for topic in os.listdir(bench_path):
        topic_path = os.path.join(bench_path, topic)
        for file in os.listdir(topic_path):
            if not file.endswith(".jsonl"): continue
            print(f"file = {file}")
            if not os.path.exists(os.path.join(target_path, topic)):
                os.makedirs(os.path.join(target_path, topic))
            target_jsonl_path = os.path.join(target_path, topic, os.path.splitext(file)[0] + ".jsonl")
            
            filter_datalist = []
            with open(os.path.join(topic_path, file), "r") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    method_code = data["method_code"]
                    func_name = data['method_name']
                    bodyline, _ = calc_fun_body_line(method_code, func_name)
                    # print(f"[method_code] =\n{method_code}")
                    print(f"func_name = {func_name}, bodyline = {bodyline}, totalline = {len(method_code)}")
                    max_bodyline = max(max_bodyline, bodyline)
                    max_totalline = max(max_totalline, len(method_code))
                    if bodyline >= 3-1 and bodyline <= 100-1: # There is a line for function declaration.
                        if len(method_code) <= 6000: # Discard excessively long data
                            print(f"filter pass !")
                            filter_datalist.append(data)
                            continue
                    print(f"filter fail !")
                    
            if len(filter_datalist) > 0:
                with open(target_jsonl_path, "w") as f:
                    for data in filter_datalist:
                        f.write(json.dumps(data) + "\n")
    print("filter done !")
    print(f"max_bodyline = {max_bodyline}")
    print(f"max_totalline = {max_totalline}")