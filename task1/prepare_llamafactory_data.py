import pandas as pd, json, argparse, os, random

# 输入 CSV: 需要列 final_text, sentiment (0/1/2)
# 输出 JSON: LLaMA-Factory 的 SFT 格式（instruction/input/output）

TEMPLATE = (
    "你是中文电商评论情感分类器。"
    "任务：判断评论情感，并且只输出一个选项字母。"
    "若无法判断，统一选择 B。"
    "\n选项：\nA. 负面\nB. 中性\nC. 正面\n"
)

def row_to_sample(text, y):
    # 训练直接让模型学会输出 A/B/C（和我们评测脚本严格对齐）
    label = {0:"A", 1:"B", 2:"C"}[int(y)]
    return {
        "instruction": TEMPLATE,
        "input": f"评论：{str(text)}\n请只输出 A 或 B 或 C。",
        "output": label,
        "history": []
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    assert "final_text" in df.columns and "sentiment" in df.columns

    df = df.dropna(subset=["final_text", "sentiment"]).reset_index(drop=True)
    data = [row_to_sample(t, y) for t, y in zip(df["final_text"], df["sentiment"])]

    random.seed(args.seed)
    random.shuffle(data)
    n = len(data); n_val = int(n * args.val_ratio)
    val = data[:n_val]; train = data[n_val:]

    with open(os.path.join(args.out_dir, "sentiment_train.json"), "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "sentiment_val.json"), "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # 生成 dataset_info.json 片段（直接可粘到 LLaMA-Factory 的 data/dataset_info.json 里）
    ds_info = {
        "sentiment_ali": {
            "file_name": "sentiment_train.json",
            "columns": {"instruction": "instruction", "input": "input", "output": "output", "history": "history"}
        },
        "sentiment_ali_val": {
            "file_name": "sentiment_val.json",
            "columns": {"instruction": "instruction", "input": "input", "output": "output", "history": "history"}
        }
    }
    with open(os.path.join(args.out_dir, "dataset_info_snippet.json"), "w", encoding="utf-8") as f:
        json.dump(ds_info, f, ensure_ascii=False, indent=2)

    print(f"OK. Train/Val = {len(train)}/{len(val)}")
    print(f"Add these entries into LLaMA-Factory/data/dataset_info.json:")
    print(json.dumps(ds_info, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
