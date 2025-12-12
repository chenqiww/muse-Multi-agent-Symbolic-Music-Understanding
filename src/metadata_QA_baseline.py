from openai import OpenAI, APIConnectionError, APITimeoutError
from inference_auth_token import get_access_token
import pandas as pd
from scipy.stats import kendalltau
import re
# model_name = "google/gemma-3-27b-it"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "openai/gpt-oss-20b"
df = pd.read_csv("data/Metadata_QA_cleaned.csv")


client = OpenAI(
    api_key=get_access_token(),
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)


def extract_option_index(pred_raw, num_options=10):

    if pred_raw is None:
        return ""

    text = pred_raw.strip()

    # 1) catch "**2." format
    m = re.search(r"\*\*(\d+)\s*[.)]", text)
    if m:
        return m.group(1)

    # 2) catch "**2**" format
    m = re.search(r"\*\*(\d+)\*\*", text)
    if m:
        return m.group(1)

    # 3) catch "2."、"2)"、"2 -" format
    m = re.search(r"\b(\d+)\s*[.)-]", text)
    if m:
        return m.group(1)

    # 4) catch simple number
    m = re.search(r"\b(\d+)\b", text)
    if m:
        idx = m.group(1)
        # ensure idx is smaller than option numbers
        if int(idx) < num_options:
            return idx

    # 5) fallback（not found)
    return ""


predictions = []
taus = []
correct = 0

for i, row in df.iterrows():
    prompt = row["prompt"] 

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        pred = response.choices[0].message.content
        pred = extract_option_index(pred)

    except Exception as e:
        print("Error at sample", i, e)
        pred = ""

    predictions.append(pred)

    # accuracy
    if str(pred) == str(row["solution"]):
        correct += 1


    print(f"[{i}] GT={row['solution']} | Pred={pred}")


accuracy = correct / len(df)

print("\n===========================")
print(f"Model: {model_name}")
print(f"Accuracy: {accuracy:.4f}")

df["pred"] = predictions

# 写出 CSV
output_path = "metadata_QA_meta_baseline_results.csv"
df.to_csv(output_path, index=False)

print(f"Saved predictions to {output_path}")