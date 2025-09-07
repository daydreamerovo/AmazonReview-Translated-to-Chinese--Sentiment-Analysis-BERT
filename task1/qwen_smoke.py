import os, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 用本地目录或可用的 HF 仓库名（二选一）
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-0.6B")
#MODEL_PATH = r"D:/models/Qwen3-4B-Instruct-2507"   # 或 "Qwen/Qwen3-4B-Instruct-2507"

# 示例：MODEL_PATH = r"D:\models\Qwen3-0.6B"
# 若要用 4B Instruct（存在且带 -2507 后缀）：
# MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"

def norm(s: str):
    s = (s or "").strip().lower()
    if "负" in s or "negative" in s: return "负面"
    if "中性" in s or "neutral" in s: return "中性"
    if "正" in s or "positive" in s: return "正面"
    for k in ["负面","中性","正面"]:
        if k in s: return k
    return "中性"

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=False)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
    local_files_only=False  # 如果你已下载到本地目录可以改 True
).eval()


text = "物流很慢，包装也烂了，体验很差。"
messages = [
    {"role":"system","content":"你是中文情感分类器"},
    {"role":"user","content":"判断情感（负面/中性/正面），只输出标签：\n评论："+text}
]

# 关掉 thinking（若模型支持该开关）
ids = tok.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False
).to(model.device)

out = model.generate(ids, max_new_tokens=8, do_sample=False)
resp = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
print("RAW:", resp)
print("LABEL:", norm(resp))
