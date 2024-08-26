# DomainEval

## Environment Setup
```bash
cd DomainEval/setup

env_name="your env name"
conda create -n "$env_name" python=3.9 -y
conda activate "$env_name"
pip install -r requirements_py39.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

## Benchmark Construction
### Domain Repository Collection

Move the code repository to the path directory of the corresponding domain `{src_data}/{domain}`.

### Test-Method Matching & Selection
```bash
domain="your domain"
version="your version"
srcdata_dir="your {src_data}"

cd DomainEval
mkdir "log_${version}"
nohup python -u sandbox.py \
--domain "$domain" \
--srcdata_dir "$srcdata_dir" \
--output_dir "bench_${version}" \
> "log_${version}/result_sandbox_${domain}.txt" &
python -u codefilter.py \
--bench_dir "bench_${version}" \
> "log_${version}/result_codefilter.txt"
```
### Instruction Generation
```bash
version="your version"
nohup python -u datagenerate.py \
--eval_dir "domaineval_${version}" \
> log_${version}/result_datagenerate.txt &
```

## Dataset

The final data is in `domaineval_{your version}`.
The data is in the format of json, each line is a json object, the format is:
```json
    "method_name":,
    "full_method_name":,
    "method_path":,
    "method_code":,
    "test_code_list":[
        {"test_code":, "code_start":, "test_path":, },
    ],
    "instruction":,
    "method_code_mask":,
```

## Evaluation
First, you need include the path and name of your model in `self.model_path_dict` within `modeleval.py`
and add your model api in `get_message` within `utils/utils_chat.py` and `self.model_name_list_api` within `modeleval.py`.

```bash
model_name="your model name or std"

# set the k in pass@k
k_pass=1 # or k_pass=5

# set the version of the dataset
version="your version"
eval_dir="domaineval_${version}"

# model inference
nohup python -u modeleval.py \
-m "$model_name" \
-b "$eval_dir" \
-k "$k_pass" \
> "result_modeleval_${model_name}_pass\@${k_pass}.txt" &

# result execution and analysis
nohup python -u resultexec.py \
-m "$model_name" \
-v "$eval_dir" \
-k "$k_pass" \
> result_exec.txt &
resultexec_pid=$!
echo $resultexec_pid
wait $resultexec_pid
mkdir -p "analyseresult/pass@${k_pass}"
python resultanalyse.py \
-m "$model_name" \
-v "$eval_dir" \
-k "$k_pass" \
> "analyseresult/pass@${k_pass}/result_analyse_${model_name}.txt"
```