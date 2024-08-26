import json
import os
from utils import get_message
from typing import Dict
import re
import ast, astor
import sys
import argparse

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

class DataGenerate():
    """
    DataGenerate is a class designed to facilitate the creation of instructional data for software engineering interviews.
    It automates the process of generating detailed instructions for specific functions within given code snippets, 
    ensuring that the instructions cover functionality, input arguments, and outputs without revealing test cases. 
    The class also includes functionality to mask the bodies of specified functions within the code, 
    providing a means to obfuscate implementation details while still allowing interviewees to understand the function's interface and purpose.
    """
    def __init__(self):
        self.prompt_template = (
            "You are preparing an interview for software engineers. "
            "The interviewees are going to complete the __FUNCTION_NAME__ function. "
            "Write a clear instruction describing this function in detail, "
            "which includes the functionality, the input arguments (if any), and the outputs (if any). "
            "Do not reveal test cases. Generate the instruction with the following format:\n"
            "```\nFunctionality: ...\nInputs: ...\nOutputs: ...\n```.\n\n__CODE__\n\n"
        )
        self.INST_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
        
    def extract_inst(self, text: str) -> str:
        """
        Extracts the instruction from the given text based on a predefined pattern.

        Args:
            text (str): The input text from which the instruction needs to be extracted.

        Returns:
            str: The extracted instruction if found, otherwise None.
        """
        if not text: return None
        if not isinstance(text, str): return None
        pattern = self.INST_BLOCK_PATTERN
        match = re.findall(pattern, text, flags=re.DOTALL) # [('', 'Python Code'), ('bash', 'ssh test')]
        if not match: return None 
        else: return match[0][1]
    
    @staticmethod    
    def maskFunc(code : str, func_name : str) -> str:
        """
        Masks the body of a specified function in the given code.

        Args:
            code (str): The source code to be processed.
            func_name (str): The name of the function whose body should be masked.

        Returns:
            str: The modified source code with the function body masked.
        """
        # Parse the code into an AST
        tree = ast.parse(code)

        # Define a class to traverse the AST and modify it
        class FunctionBodyRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == func_name:
                    # Mask the body of the function
                    node.body = " [MASK]"
                return node

        # Create an instance of the modifier
        remover = FunctionBodyRemover()
        # Apply the modifier to the AST
        masked_tree = remover.visit(tree)
        # Convert the modified AST back to code
        masked_code = astor.to_source(masked_tree).strip()
        return masked_code + "\n"
    
    def generateDataPoint(self, data : Dict, try_max : int = 2, debug : bool = False):
        """
        Generates a data point by processing the given data dictionary.

        Args:
            data (Dict): A dictionary containing the method name, full method name, and method code.
            try_max (int, optional): The maximum number of attempts to generate the instruction. Defaults to 2.
            debug (bool, optional): If True, prints debug information. Defaults to False.

        Returns:
            Dict: A dictionary containing the original data along with the generated instruction and masked method code.

        Raises:
            Exception: If the instruction cannot be generated after the maximum number of attempts.
        """        
        messages = [
            {
                "role": "user",
                "content": 
                    self.prompt_template
                        .replace('__FUNCTION_NAME__', data['full_method_name'])
                        .replace('__CODE__', data["method_code"])
            }
        ]
        
        last_exception = None
        instruction = None
        
        for _ in range(try_max):
            try:
                response = get_message(
                    messages = messages, 
                    model_name = "Qwen2-72B-Instruct-GPTQ-Int4",
                    temperature = 0.8,
                    n = 1,
                    # max_tokens = 32000,
                )
                if isinstance(response, list):
                    assert len(response) == 1, "response should be a list with only one element"
                    response = response[0]
                instruction = self.extract_inst(text = response)
                if instruction : break
            except Exception as e:
                last_exception = e
        
        if instruction is None and last_exception is not None:
            raise Exception(f"{last_exception}")
        if not instruction: return None
        data["instruction"] = instruction
        data["method_code_mask"] = \
            self.maskFunc(
                code = data['method_code'], 
                func_name = data['method_name']
            )
        if debug:
            print("-"*20)
            print(f"full_method_name: {data['full_method_name']}")
            print(f"\nmethod_code:\n{data['method_code']}")
            print(f"\ninstruction:\n{instruction}")
            print(f"\nmethod_code_mask =\n{data['method_code_mask']}")
        return data

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str, default="domaineval_20240711")
    args = parser.parse_args()
    
    bench_path = os.path.join(os.getcwd(), "filter")
    target_path = os.path.join(args.eval_dir)
    
    data_generate = DataGenerate()
    
    for topic in os.listdir(bench_path):
        topic_path = os.path.join(bench_path, topic)
        if not os.path.isdir(topic_path): continue
        for file in os.listdir(topic_path):
            if not file.endswith(".jsonl"): continue
            print(f"Processing topic: {topic}/ file: {file}")
            if not os.path.exists(os.path.join(target_path, topic)):
                os.makedirs(os.path.join(target_path, topic))
            target_jsonl_path = os.path.join(target_path, topic, os.path.splitext(file)[0] + ".jsonl")
            if os.path.exists(target_jsonl_path):
                print(f"topic: {topic}/ file: {file} already exists") 
                continue
            with open(target_jsonl_path, "w") as f:
                pass
            
            with open(os.path.join(topic_path, file), "r", encoding = 'utf-8') as f:
                lines = f.readlines()
            data_list = [json.loads(line) for line in lines]
            
            for d in data_list:
                # if not d["full_method_name"] == "phase_retarder": continue
                try:
                    data = data_generate.generateDataPoint(d, debug=False)
                except Exception as e:
                    print(f"full_method_name: {d['full_method_name']}\n{e}")
                    continue
                if data:
                    with open(target_jsonl_path, "a", encoding = 'utf-8') as f:
                        f.write(json.dumps(data) + "\n")
                else:
                    print(f"full_method_name: {d['full_method_name']}")
                    print("Instruction is None")
            print(f"Complete topic: {topic}/ file: {file}")
            
    print("Data Generate Completed")