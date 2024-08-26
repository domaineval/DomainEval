domain="Basic"
domain="Computation"
domain="Cryptography"
domain="Visualization"
domain="Network"
domain="System"

version="20240711"
srcdata_dir="../srcdata"

mkdir "log_${version}"
nohup python -u sandbox.py \
--domain "$domain" \
--srcdata_dir "$srcdata_dir" \
--output_dir "bench_${version}" \
> "log_${version}/result_sandbox_${domain}.txt" &

python -u codefilter.py > log_${version}/result_codefilter.txt
nohup python -u datagenerate.py > log_${version}/result_datagenerate_0726.txt &

# 参数设置
model_name="deepseek-coder-6.7b-instruct"
model_name="DeepSeek-Coder-V2-Lite-Instruct"
model_name="deepseek-coder-33b-instruct"
model_name="CodeLlama-7b-Instruct-hf"
model_name="CodeLlama-13b-Instruct-hf"
model_name="CodeLlama-34b-Instruct-hf"
model_name="Llama-2-13b-chat-hf"
model_name="Phi-3-medium-4k-instruct"
model_name="CodeQwen1.5-7B-Chat"
model_name="Qwen2-72B-Instruct-GPTQ-Int4"
model_name="gpt-4o-mini"
model_name="gpt-3.5-turbo"
model_name="std"

k_pass=1
k_pass=5

# 模型推理
nohup python -u modeleval.py -m "$model_name" -k "$k_pass" > "result_modeleval_${model_name}_pass\@${k_pass}.txt" &

nohup python -u resultexec.py -m "$model_name" -k "$k_pass" > result_exec.txt &

resultexec_pid=$!
echo $resultexec_pid
wait $resultexec_pid
python resultanalyse.py -m "$model_name" -k "$k_pass" > "analyseresult/pass@${k_pass}/result_analyse_${model_name}.txt"