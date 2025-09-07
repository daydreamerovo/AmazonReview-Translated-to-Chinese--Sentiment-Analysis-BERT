import argparse
import os
import time
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    logging,
)

# ── 安静一点（保留我们手动打印的进度） ──
logging.set_verbosity_error()

# ── 你的数据列 ──
TEXT_COL = "final_text"
LABEL_COL = "sentiment"   # 0=负面, 1=中性, 2=正面

ID2STR = {0: "负面", 1: "中性", 2: "正面"}
STR2ID = {v: k for k, v in ID2STR.items()}
CHOICE2ID = {"A": 0, "B": 1, "C": 2}


def normalize_choice(s: str) -> int:
    """只接受 A/B/C（首字符），兜底再做一次粗略判断。"""
    s = (s or "").strip()
    for ch in s:
        if not ch.isspace():
            ch = ch.upper()
            if ch in CHOICE2ID:
                return CHOICE2ID[ch]
            break
    t = s.lower()
    if any(k in t for k in ["负", "差", "垃圾", "退货", "bad", "negative"]): return 0
    if any(k in t for k in ["中性", "一般", "还行", "neutral"]): return 1
    if any(k in t for k in ["正", "好评", "满意", "喜欢", "推荐", "good", "positive"]): return 2
    return 1


def build_zeroshot_prompt(text: str) -> str:
    return (
        "你是中文电商评论情感分类器。\n"
        "任务：判断评论情感，并且**只输出一个选项字母**。\n"
        "选项：\nA. 负面\nB. 中性\nC. 正面\n\n"
        f"评论：{text}\n"
        "请只输出 A 或 B 或 C。不要输出其他任何内容。"
    )


def build_fewshot_prompt(text: str, shots) -> str:
    header = (
        "你是中文电商评论情感分类器。\n"
        "根据示例学习，然后**只输出一个选项字母**。\n"
        "选项：\nA. 负面\nB. 中性\nC. 正面\n\n【示例】\n"
    )
    demo = ""
    for t, y in shots:
        demo += f"评论：{t}\n答案：{'ABC'[int(y)]}\n\n"
    query = f"【待判断】\n评论：{text}\n请只输出 A 或 B 或 C。"
    return header + demo + query


def load_model(model_id_or_dir: str):
    """自动识别本地目录/仓库名；设置干净的生成配置。"""
    is_local = os.path.isdir(model_id_or_dir)
    tok = AutoTokenizer.from_pretrained(
        model_id_or_dir, trust_remote_code=True, local_files_only=is_local
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        local_files_only=is_local,
    ).eval()

    gc = GenerationConfig.from_model_config(model.config)
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    model.generation_config = gc
    return tok, model


def eval_model(model_id_or_dir: str, df: pd.DataFrame, fewshot_k=0, max_samples=None):
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    tok, model = load_model(model_id_or_dir)

    # few-shot 示例池（每类各取 k 条）
    shots = None
    if fewshot_k > 0:
        shots = []
        for c in [0, 1, 2]:
            sub = df[df[LABEL_COL] == c]
            take = min(fewshot_k, len(sub))
            shots.extend(list(zip(sub[TEXT_COL].head(take).tolist(),
                                  sub[LABEL_COL].head(take).tolist())))

    preds, labels, total = [], [], 0.0
    N = len(df)

    for i, r in df.iterrows():
        text = str(r[TEXT_COL])[:1500]  # 稍短一点更稳
        gold = int(r[LABEL_COL])
        prompt = build_fewshot_prompt(text, shots) if shots else build_zeroshot_prompt(text)

        messages = [
            {"role": "system", "content": "你必须严格按要求输出。"},
            {"role": "user", "content": prompt},
        ]
        # 有些型号支持 enable_thinking；不支持就回退
        try:
            ids = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False
            ).to(model.device)
        except TypeError:
            ids = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

        attention_mask = torch.ones_like(ids)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                ids,
                attention_mask=attention_mask,
                max_new_tokens=2,   # 强制仅输出 A/B/C
                do_sample=False,
            )
        total += time.time() - t0

        txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        pred = normalize_choice(txt)
        preds.append(pred)
        labels.append(gold)

        if (i + 1) % 50 == 0 or (i + 1) == N:
            print(f"{i+1}/{N} done, last_time={total / (i+1):.3f}s(avg so far)")

    acc = (pd.Series(preds) == pd.Series(labels)).mean()
    avg_time = total / N
    return acc, avg_time, N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="包含 final_text,sentiment 的 CSV")
    parser.add_argument("--model_dirs", nargs="+", required=True, help="本地模型目录或 HF 仓库名（可多个）")
    parser.add_argument("--fewshot_k", type=int, default=0, help="每类示例数；0=zero-shot")
    parser.add_argument("--max_samples", type=int, default=None, help="抽样数量")
    # 结果追加/标记
    parser.add_argument("--results_path", type=str, default="results_qwen_eval.csv", help="结果CSV路径")
    parser.add_argument("--append", action="store_true", help="若存在则在文件末尾追加而非覆盖")
    parser.add_argument("--run_name", type=str, default=None, help="本轮评测标记名（可选）")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    assert TEXT_COL in df.columns and LABEL_COL in df.columns, f"需要列：{TEXT_COL},{LABEL_COL}"

    rows = []
    for model_dir in args.model_dirs:
        try:
            acc, avg_t, n_used = eval_model(
                model_dir, df, fewshot_k=args.fewshot_k, max_samples=args.max_samples
            )
            print(f"[{model_dir}] shots={args.fewshot_k}  acc={acc:.4f}  avg_time={avg_t:.3f}s")
            rows.append({
                "model_dir": model_dir,
                "shots": args.fewshot_k,
                "accuracy": acc,
                "avg_time_s": avg_t,
                "n_samples": n_used,
                "run_name": args.run_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            print(f"[{model_dir}] ERROR: {e}")
            rows.append({
                "model_dir": model_dir,
                "shots": args.fewshot_k,
                "accuracy": None,
                "avg_time_s": None,
                "error": str(e),
                "n_samples": len(df) if args.max_samples is None else args.max_samples,
                "run_name": args.run_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

    # 追加/覆盖保存
    out_path = args.results_path
    new_df = pd.DataFrame(rows)
    if args.append and os.path.exists(out_path):
        old_df = pd.read_csv(out_path)
        all_cols = sorted(set(old_df.columns).union(new_df.columns))
        old_df = old_df.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  (rows={len(combined)})")


if __name__ == "__main__":
    main()
