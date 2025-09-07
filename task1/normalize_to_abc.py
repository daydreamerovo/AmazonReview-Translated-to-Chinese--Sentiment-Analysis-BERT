import json, argparse, os, tempfile

def to_abc(v:str):
    s = str(v).strip()
    if s in {"A","B","C"}: return s
    if s in {"0","1","2"}: return {"0":"A","1":"B","2":"C"}[s]
    sl = s.lower()
    if any(k in s for k in ["负","差","差评","垃圾","退货"]) or "neg" in sl or "bad" in sl: return "A"
    if any(k in s for k in ["中","一般","还行","普通"]) or "neu" in sl or "neutral" in sl: return "B"
    if any(k in s for k in ["正","好评","满意","喜欢","推荐","很好","棒"]) or "pos" in sl or "good" in sl: return "C"
    return "B"

def norm_file(path_in, path_out):
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path_out), suffix=".jsonl").name
    with open(path_in, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            obj = json.loads(line)
            # 取出可能的标签字段并转换为 A/B/C
            y = obj.get("output", obj.get("label", obj.get("sentiment", obj.get("target"))))
            y = to_abc(y)
            inst = obj.get("instruction") or "你是中文电商评论情感分类器。只输出一个选项字母：A=负面，B=中性，C=正面。"
            inp  = obj.get("input") or obj.get("text") or obj.get("final_text") or ""
            hist = obj.get("history", [])
            rec = {"instruction": inst, "input": inp, "output": y, "history": hist}
            fout.write(json.dumps(rec, ensure_ascii=False)+"\n")
    os.replace(tmp, path_out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()
    norm_file(args.in_path, args.out_path)
    print("OK:", args.out_path)
