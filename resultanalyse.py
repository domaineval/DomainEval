import os
import json
import argparse
import numpy as np
from datetime import datetime

def pass_at_k(n : int, c : int , k : int) -> float:
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """ 
    if n - c < k : return 1.0 
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Result Analyse")
    parser.add_argument("--model_name", "-m", type=str, default="std", help="Model Name")
    parser.add_argument("--result_root_dir", "-r", type=str, default="executeresult", help="Result Directory")
    parser.add_argument("--version", "-v", type=str, default="domaineval_20240711", help="Version")
    parser.add_argument("--k_pass", "-k", type=int, default=1, help="pass@k")
    args = parser.parse_args()
    
    result_dir = os.path.join(args.result_root_dir, args.version, args.model_name, f"pass_{args.k_pass}")
    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"Result Directory {result_dir} Not Found")
    if args.k_pass < 1: raise ValueError("k in pass@k must be greater than 0")
    
    print(f"Result Directory: {result_dir}")
    print(f"Model: {args.model_name}")
    
    now = datetime.now()
    print("Time: ", now.strftime("%Y-%m-%d %H:%M:%S"))
    
    result_dict = {}
    result_example = {}
    for topic in os.listdir(result_dir):
        topic_path = os.path.join(result_dir, topic)
        if not os.path.isdir(topic_path): continue
        
        print(f"\nAnalyse topic: {topic}")
        topic_all = 0
        topic_acc = 0
        
        for file in os.listdir(topic_path):
            if not file.endswith(".jsonl"): continue
            # print(f"Analyse {topic}/ {file}")
            
            result_jsonl_path = os.path.join(topic_path, file)
            with open(result_jsonl_path, "r", encoding = 'utf-8') as f:
                lines = f.readlines()
            data_list = [json.loads(line) for line in lines]
            
            if len(data_list) <= 0: continue
            # Sort the out of order results
            data_list.sort(key = lambda x: (x["model_name"], x["full_method_name"], x["method_path"], x["method_code_mask"]))
            
            # Calculation the maximum value of k in pass@k
            repeat_K = 0
            while repeat_K < len(data_list) and \
                  data_list[0]["model_name"] == data_list[repeat_K]["model_name"] and \
                  data_list[0]["full_method_name"] == data_list[repeat_K]["full_method_name"]:
                    if "method_path" in data_list[0] and "method_path" in data_list[repeat_K]:
                        if data_list[0]["method_path"] != data_list[repeat_K]["method_path"]:
                            break
                    if "method_code_mask" in data_list[0] and "method_code_mask" in data_list[repeat_K]:
                        if data_list[0]["method_code_mask"] != data_list[repeat_K]["method_code_mask"]:
                            break
                    repeat_K += 1
            repeat_K = args.k_pass
            assert repeat_K >= args.k_pass
            assert len(data_list) % repeat_K == 0, f"file = {file}\nlen(data_list) = {len(data_list)}, repeat_K = {repeat_K}"
            
            repo_all = 0
            repo_acc = 0
            # repeat_K == args.k_pass
            for i in range(0, len(data_list), repeat_K):
                one_data_group = data_list[i:i + args.k_pass]
                pass_flag = False
                for d in one_data_group:
                    assert d["model_name"] == args.model_name
                    if d["result"] == "passed": pass_flag = True
                    if d["result"] in result_example:
                        result_example[d["result"]] += 1
                    else: result_example[d["result"]] = 1
                repo_all += 1
                if pass_flag is True: repo_acc += 1
                
            print(f"{topic}/{os.path.splitext(file)[0]} pass@{args.k_pass} = {repo_acc} / {repo_all} = {repo_acc / repo_all * 100:.2f}%")
            
            topic_all += repo_all
            topic_acc += repo_acc
            
        print(f"{topic} pass@{args.k_pass} = {topic_acc} / {topic_all} = {topic_acc / topic_all * 100:.2f}%")
        result_dict[topic] = {"all": topic_all, "acc": topic_acc}
    
    print()
    test_all = 0
    test_acc = 0
    excel_result = []
    for topic in result_dict:
        test_all += result_dict[topic]["all"]
        test_acc += result_dict[topic]["acc"]
        print(f"{topic} pass@{args.k_pass} = {result_dict[topic]['acc']} / {result_dict[topic]['all']} = {result_dict[topic]['acc'] / result_dict[topic]['all'] * 100:.2f}%")
        excel_result.append(result_dict[topic]['acc'] / result_dict[topic]['all'] * 100)
    print(f"Total pass@{args.k_pass} = {test_acc} / {test_all} = {test_acc / test_all * 100:.2f}%")
    print(f"Average pass@{args.k_pass} = {sum(excel_result)/len(excel_result):.2f}%")
    excel_result.append(test_acc / test_all * 100)
    excel_result.append(sum(excel_result[:-1])/len(excel_result[:-1]))
    for r in excel_result: print(f"{r:.2f}", end='\t')
    print()
    print()
    result_example = dict(sorted(result_example.items(), key=lambda item: item[1], reverse=True))
    for r in result_example:
        print(f"{r}: {result_example[r]}")