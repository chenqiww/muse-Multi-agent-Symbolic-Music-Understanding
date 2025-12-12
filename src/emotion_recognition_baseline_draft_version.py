from openai import OpenAI, APIConnectionError, APITimeoutError
from inference_auth_token import get_access_token
import pandas as pd
from scipy.stats import kendalltau
import re
import sys

# model_name = "google/gemma-3-27b-it"
model_name = "openai/gpt-oss-20b"
#model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
df = pd.read_csv("data/Emotion_Recognition_cleaned.csv")

if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
    n = int(sys.argv[1][2:])
    df = df.head(n)

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
raw_responses = []
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

        raw_response = response.choices[0].message.content
        pred = extract_option_index(raw_response)
        raw_responses.append(raw_response)

    except Exception as e:
        print("Error at sample", i, e)
        pred = ""
        raw_responses.append("")

    predictions.append(pred)

    # accuracy
    if str(pred) == str(row["solution"]):
        correct += 1

    print(f"[{i}] GT={row['solution']} | Pred={pred}")

# Save results to CSV
results_df = pd.DataFrame({
    'index': df.index,
    'ground_truth': df['solution'].values,
    'prediction': predictions,
    'raw_response': raw_responses
})
results_df.to_csv('emotion_recognition_results.csv', index=False)

accuracy = correct / len(df)

print("\n===========================")
print(f"Model: {model_name}")
print(f"Accuracy: {accuracy:.4f}")

