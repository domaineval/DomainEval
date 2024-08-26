import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
from typing import List, Dict
from tqdm import tqdm
from utils import get_message
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

import sys

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

class ModelEval():
    def __init__(self, model_name : str):
        self.prompt_template = (
            "You are a software engineering interviewee. "
            "You need to complete the __FUNCTION_NAME__ function according to the instruction, "
            "which includes the functionality, the input arguments (if any), and the outputs (if any).\n"
            "You need to complete based on the provided code, and the areas that need to be completed have already been masked by \"[MASK]\".\n"
            "You can only use the provided import statements as the runtime environment.\n"
            "You should copy the input code with no omission, and complete the masked areas with the correct code.\n"
            "You should answer a complete function includes context and import. "
            "Output the completed code with the format: ```python ```\n"
            "\n**Instruction**\n__INSTRUCTION__\n"
            "\n**CODE**\n__CODE__\n"
        )
        self.CODE_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
        
        self.model_path_dict = {
            "deepseek-coder-6.7b-instruct" : "your deepseek-coder-6.7b-instruct",
            "DeepSeek-Coder-V2-Lite-Instruct" : "your DeepSeek-Coder-V2-Lite-Instruct",
            "deepseek-coder-33b-instruct" : "your deepseek-coder-33b-instruct",
            "CodeLlama-7b-Instruct-hf" : "your CodeLlama-7b-Instruct-hf",
            "CodeLlama-13b-Instruct-hf" : "your CodeLlama-13b-Instruct-hf",
            "CodeLlama-34b-Instruct-hf" : "your CodeLlama-34b-Instruct-hf",
            "Llama-2-13b-chat-hf" : "your Llama-2-13b-chat-hf",
            "Phi-3-medium-4k-instruct" : "your Phi-3-medium-4k-instruct",
            "CodeQwen1.5-7B-Chat" : "your CodeQwen1.5-7B-Chat",
        }
        self.model_name_list_api = ["gpt-3.5-turbo", "Qwen2-72B-Instruct-GPTQ-Int4", "gpt-4o-mini"]
        self.model_name_list_load = list(self.model_path_dict.keys())
        
        if model_name not in self.model_name_list_api + self.model_name_list_load + ["std"]: 
            raise ValueError(f"{model_name} Not Supported")
        
        # load model
        self.model_name = model_name
        if self.model_name in self.model_name_list_load:
            self.model_name_or_path = self.model_path_dict[self.model_name]
            if os.path.exists(self.model_name_or_path):
                print(f"Loading model from {self.model_name_or_path}")
            else: raise FileNotFoundError(f"{self.model_name_or_path} Not Found")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"device = {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)
            self.model.eval()
            print(f"Model {self.model_name} Loaded")
            print(f"{self.model.config}")
        
        
    def extract_code(self, text: str) -> str:
        """
        Extracts the code from a given text based on a predefined pattern.

        Parameters:
        - text (str): The input text from which the code block is to be extracted.

        Returns:
        - str: The extracted code if a match is found, otherwise None.
        """
        if not text: return None
        if not isinstance(text, str): return None
        pattern = self.CODE_BLOCK_PATTERN
        match = re.findall(pattern, text, flags=re.DOTALL) # [('', 'Python Code'), ('bash', 'ssh test')]
        if not match: return None 
        else: return match[0][1]
    
    @torch.no_grad()
    def eval_generate(
            self,
            datalist : List[Dict],
            k_pass : int = 1,
            debug : bool = False,
            logging : bool = True,
            topic_name = None,
            file_name = None,
            result_base_path = None,
        ) -> List[Dict]:
        
        if debug: print(f"len = {len(datalist)} model_name = {self.model_name}")
        if logging is True:
            if topic_name is None or file_name is None or result_base_path is None:
                raise ValueError("Warning: No record path specified !")
            else:
                test_result_path = os.path.join(result_base_path, self.model_name, f"pass_{k_pass}",topic)
                if not os.path.exists(test_result_path): os.makedirs(test_result_path)
                test_result_path = os.path.join(test_result_path, file_name)
                if os.path.exists(test_result_path):
                    print(f"{test_result_path} already exists")
                    return []
        
        result_list = []
        for data in tqdm(datalist):
            messages = [
                {
                    "role": "user",
                    "content": 
                        self.prompt_template
                            .replace('__FUNCTION_NAME__', data['full_method_name'])
                            .replace('__CODE__', data["method_code_mask"])
                            .replace('__INSTRUCTION__', data["instruction"])
                }
            ]
            
            # pass@1, set temperature to 0.0
            # pass@5, set temperature to 0.2
            if k_pass == 1: temperature = 0.0
            elif k_pass == 5: temperature = 0.2
            else: temperature = 0.8

            # inference pass@k
            if self.model_name == "std": 
                code_model_response = [data["method_code"]]
                if debug:
                    print("-"*20)
                    print(f"user:\n{messages[0]['content']}") 
                    print(f"\nmodel:\n{code_model_response[0]}")
            elif self.model_name in self.model_name_list_api:
                try:
                    if debug: print(f"len(message) = {len(messages[0]['content'])}")
                    response = get_message(
                        messages = messages, 
                        model_name = self.model_name,
                        temperature = temperature,
                        max_tokens = 2048,
                        n = k_pass,
                        debug = False,
                    )
                except ConnectionError as e:
                    print(f"Connection error: {e}")
                    response = "Connection error"
                    return []
                if debug:
                    print("-"*20)
                    print(f"user:\n{messages[0]['content']}")
                    for i in range(k_pass):
                        print(f"\nmodel[{i}]:\n{response[i]}")
                code_model_response = [self.extract_code(r) for r in response]
            elif self.model_name in self.model_name_list_load:
                model_chat_flag = any(e in self.model_name for e in ["instruct", "Instruct", "chat", "Chat"])
                if model_chat_flag:
                    inputs = self.tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    ).to(self.model.device)
                    # tokenizer.eos_token_id is the id of <|EOT|> token
                else: 
                    inputs = self.tokenizer.encode(
                        messages[0]['content'], 
                        return_tensors="pt"
                    ).to(self.model.device)
                if debug:
                    # print(f"type(inputs)={type(inputs)}")
                    print(f"\ninputs.shape={inputs.shape}")

                if inputs.shape[1] > 5000: continue
                try:
                    if temperature > 0.0:
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens = 2048, 
                            do_sample=True,
                            temperature = temperature, 
                            top_k=50, 
                            top_p=0.95,
                            num_return_sequences=k_pass,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        outputs = self.model.generate(
                            inputs,
                            max_new_tokens = 2048, 
                            do_sample=False,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    response = []
                    for i in range(k_pass):
                        response.append(self.tokenizer.decode(outputs[i][len(inputs[0]):], skip_special_tokens=True))
                except Exception as e:
                    response = [f"Error: {e}"] * k_pass
                    print(response)
                if debug:
                    print("-"*20)
                    print(f"user:\n{messages[0]['content']}")
                    for i in range(k_pass):
                        print(f"\nmodel[{i}]:\n{response[i]}")
                code_model_response = response
                if model_chat_flag: 
                    code_model_response = [self.extract_code(r) for r in response]
            else: raise ValueError(f"{self.model_name} Not Supported")
            
            assert type(code_model_response) == list
            assert len(code_model_response) == k_pass
            for i in range(k_pass):
                result_dict = {
                    "model_name": self.model_name,
                    "repository": os.path.splitext(file)[0],
                    "full_method_name": data["full_method_name"],
                    "method_path": data["method_path"],
                    "response": code_model_response[i],
                    "test_code_list": data["test_code_list"],
                    "method_code_mask": data["method_code_mask"],
                }
                result_list.append(result_dict)
        
        if logging is True:
            with open(test_result_path, "w", encoding = 'utf-8') as f:
                for result in result_list:
                    f.write(json.dumps(result) + "\n")
        return result_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Eval_Generate")
    parser.add_argument("--model_name", "-m", type=str, default="std", help="Model Name")
    parser.add_argument("--k_pass", "-k", type=int, default=1, help="Pass@K")
    parser.add_argument("--benchmark_dir", "-b", type=str, default="domaineval_20240711", help="Benchmark Directory")
    parser.add_argument("--output_root_dir", "-o", type=str, default="modelresult", help="Output Root Directory")
    args = parser.parse_args()
    
    doceb_path = os.path.join(os.getcwd(), args.benchmark_dir)
    result_path = os.path.join(os.getcwd(), args.output_root_dir, args.benchmark_dir)
    if not os.path.exists(result_path): os.makedirs(result_path)
    
    evaluator = ModelEval(model_name = args.model_name)
    print(f"model_name : {args.model_name}")
    print(f"pass@k : pass@{args.k_pass}")
    print(f"from : {doceb_path}")
    print(f"to {result_path}")
    for topic in os.listdir(doceb_path):
        topic_path = os.path.join(doceb_path, topic)
        if not os.path.isdir(topic_path): continue
        for file in os.listdir(topic_path):
            if not file.endswith(".jsonl"): continue
            print(f"Eval_Generate topic: {topic}/ file: {file}")
            
            # if file == "sympy.jsonl": continue
            
            datalist = []
            with open(os.path.join(topic_path, file), "r", encoding = 'utf-8') as f:
                for line in f:
                    datalist.append(json.loads(line))

            if len(datalist) <= 0: continue
            evaluator.eval_generate(
                datalist = datalist,
                k_pass = args.k_pass,
                debug = False,
                logging = True,
                topic_name = topic,
                file_name = file,
                result_base_path = result_path,
            )
            
            print(f"Model eval_generate topic: {topic}/ file: {file} completed")
            
    print("Eval_Generate End")