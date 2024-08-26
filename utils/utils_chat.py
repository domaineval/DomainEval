import json
import requests
import time
from typing import List, Dict

def get_message(
    messages : List[Dict], 
    model_name : str = "gpt-3.5-turbo",
    temperature : float = None,
    n : int = 1,
    max_tokens : int = None,
    max_new_tokens : int = None,
    stream : bool = False,
    debug : bool = False
    ):
    model_name_list_api = [
        "gpt-3.5-turbo", 
        "Qwen2-72B-Instruct-GPTQ-Int4", 
        "gpt-4o-mini",
        "gpt-4o", 
        "gpt-4-turbo", 
        "gpt-4",
    ]
    if model_name not in model_name_list_api: 
        raise ValueError(f"{model_name} Not Supported")
    
    if model_name in model_name_list_api:
        headers = {
            "Content-Type": "application/json"
        }
        if model_name in ["gpt-3.5-turbo", "gpt-4o-mini"]:
            url = "https://api.openai.com/v1/chat/completions"
            API_KEY = "your API_KEY"
            headers["Authorization"] = f"Bearer {API_KEY}"
        if model_name == "Qwen2-72B-Instruct-GPTQ-Int4":
            url = "your url"
        
        data = {
            "model": model_name,
            "messages": messages,
            "n" : n,
            "stream": stream,
        }
        if temperature is not None:
            data["temperature"] = temperature
        # pass@1, set temperature to 0.0
        # pass@5, set temperature to 0.2
        
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if max_new_tokens is not None:
            data["max_new_tokens"] = max_new_tokens
        
        if debug : print(f"data={data}")
        while True:
            try:
                response = requests.post(url, json=data, headers=headers)
                break
            except Exception as e:
                print(f"\nError: {e}")
                time.sleep(10)
        if debug : print(f"response={response.text}")

        message = json.loads(response.text)
        if not isinstance(message, Dict) or 'choices' not in message:
            repeat_index = 0
            while not isinstance(message, Dict) or 'choices' not in message:
                if repeat_index > 5 : raise ConnectionError(f"{url} Error\nmessage=\n{message}")
                if debug : print(f"message=\n{message}")
                time.sleep(5)
                response = requests.post(url, json=data, headers=headers)
                if debug : print(f"response=\n{response.text}")
                message = json.loads(response.text)
                repeat_index += 1
            if not isinstance(message, Dict) or 'choices' not in message: 
                raise ConnectionError(f"{url} Error\nmessage=\n{message}")
        if len(message['choices']) != n:
            raise ValueError(f"{model_name} response num error")
        if "gpt" in model_name: time.sleep(1)
        return [message['choices'][i]['message']['content'] for i in range(n)]

if __name__ == "__main__":
    content = \
"""
Who are you?
"""
    messages = [{"role": "user", "content": content}]
    n = 3
    model_message = get_message(
        messages = messages, 
        model_name = "Qwen2-72B-Instruct-GPTQ-Int4",
        # model_name = "gpt-4o-mini",
        temperature = 0.8,
        n = n,
        max_tokens = 20,
        debug = True,
    )
    messages.append({"role": "assistant", "content": model_message})
    for i in range(n):
        print("-"*20)
        print(model_message[i])
    with open("result_chat.txt", "w", encoding="utf-8") as f:
        for message in messages:
            if message["role"] == "user":
                f.write(f"{message['role']}:\n{message['content']}\n\n")
            else:
                for i in range(n):
                    f.write(f"{message['role']}[{i}]:\n{message['content'][i]}\n\n")