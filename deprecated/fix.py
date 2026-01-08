import re
import pandas as pd
from pathlib import Path

def parse_raw_json(raw):
    if not raw or pd.isna(raw):
        return {"answer": None, "confidence": None, "rationale": None}

    raw = str(raw)

    # Step 1: Extract the 'raw' value if it's a dict-like wrapper
    raw_match = re.search(r"'raw':\s*(.*)", raw, re.DOTALL)
    if raw_match:
        raw = raw_match.group(1).strip()
        # Remove trailing comma or closing brace if present
        raw = re.sub(r"[},]\s*$", "", raw)

    # Step 2: Remove ```json or ```
    raw = re.sub(r"```(?:json)?", "", raw)

    # Step 3: Normalize whitespace and line breaks
    raw = raw.replace("\n", " ").replace("\t", " ").strip()

    # Step 4: Convert single quotes to double quotes for simple key/value parsing
    raw = raw.replace("'", '"')

    # Step 5: Try to extract fields manually using regex
    def extract_field(name):
        pattern = rf'"{name}"\s*:\s*"([^"]*?)"'
        match = re.search(pattern, raw, re.DOTALL)
        return match.group(1).strip() if match else None

    answer = extract_field("answer")
    confidence = extract_field("confidence")
    rationale = extract_field("rationale")

    # Step 6: Try to parse confidence as float
    try:
        confidence = float(confidence) if confidence is not None else None
    except:
        confidence = None

    return {
        "answer": answer,
        "confidence": confidence,
        "rationale": rationale
    }

models = ["llava-v1.6-mistral-7b-hf", "Qwen2.5-VL-7B-Instruct", "InternVL3-8B"]
datasets = ["cvr", "bp", "marsvqa", "raven"]
vers = ["ver1", "ver2", "ver3"]
strategies = ["classification", "direct", "contrastive", "descriptive"]
results = ["results", "evaluation_results", "all_results_concat"]

for dataset in datasets:
    for strategy in strategies:
        for model in models:
            for ver in vers:
                for result in results:
                    if result == "all_results_concat":
                        path = f"../results/{result}.csv"
                    else:
                        path = f"../results/{dataset}/{strategy}/{model}/{ver}/{result}.csv"
                    try:
                        df = pd.read_csv(
                            path,
                            dtype={"problem_id": str}, 
                        )
                    except FileNotFoundError:
                        print(f"File not found: {path}, trying alternative path.")
                        continue

                    df["problem_id"] = df["problem_id"].str.strip()
                    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

                    if "reasoning" in df.columns and "judge_rationale" not in df.columns:
                        df.rename(columns={"reasoning": "judge_rationale"}, inplace=True)
                    
                    if result == "all_results_concat":
                        mask = df["judge_rationale"].notna() & (df["judge_rationale"].str.strip() != "")
                        df.loc[mask, "judge_model_name"] = "Mistral-7B-Instruct-v0.3"
                        df.loc[mask, "judge_model_param_set"] = 1

                    else:
                        if "judge_rationale" in df.columns and "judge_model_name" not in df.columns:
                            df["judge_model_name"] = "Mistral-7B-Instruct-v0.3"
                            
                        if "judge_rationale" in df.columns and "judge_model_param_set" not in df.columns:
                            df["judge_model_param_set"] = 1

                    mask = df["answer"].isna() | (df["answer"] == '')

                    parsed = df.loc[mask, "raw_response"].apply(parse_raw_json)

                    df.loc[mask, "answer"] = parsed.apply(lambda x: x["answer"])
                    df.loc[mask, "confidence"] = (
                        parsed.apply(lambda x: x["confidence"])
                        .astype(float)
                    )
                    df.loc[mask, "rationale"] = parsed.apply(lambda x: x["rationale"])

                    df.to_csv(path, index=False)

model_name = "Mistral-7B-Instruct-v0.3"
models = ["llava-v1.6-mistral-7b-hf", "Qwen2.5-VL-7B-Instruct", "InternVL3-8B"]
datasets = ["bp"]
vers = ["ver1", "ver2", "ver3"]
strategies = ["direct", "contrastive", "descriptive"]

target_files = {
    "evaluation_results_.csv",
    "evaluation_results_metrics.json",
    "evaluation_results_summary.json",
}

for dataset in datasets:
    for strategy in strategies:
        for model in models:
            for ver in vers:

                base_dir = Path("../results") / dataset / strategy / model / ver

                if not base_dir.exists():
                    continue

                for fname in target_files:
                    old_path = base_dir / fname
                    if not old_path.exists():
                        continue

                    new_path = old_path.with_name(
                        f"{old_path.stem}_{model_name}_1{old_path.suffix}"
                    )

                    old_path.rename(new_path)
                    print(f"Renamed: {old_path} → {new_path}")
