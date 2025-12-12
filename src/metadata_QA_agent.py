from openai import OpenAI, APIConnectionError, APITimeoutError
from inference_auth_token import get_access_token
import pandas as pd
from scipy.stats import kendalltau
import re
model_name = "google/gemma-3-27b-it"
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
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


def abc_expert_prompt(input_abc):
    return f"""
You are an ABC notation expert. Your job is to interpret the following ABC score.

Score:
{input_abc}

Explain the meaning of each ABC component in a structured and concise way.
Focus on:
- Key (K:)
- Meter / time signature (M:)
- Default note length (L:)
- Chord symbols
- Bar boundaries
- Rhythm patterns
- Melodic contour
- Tuplets and ornaments
- Phrase structure

Do NOT answer the user's question.
ONLY produce an analysis of the ABC score.
"""

def evaluator_prompt(analysis, task_prompt):
    return f"""
You are the evaluator agent.

You will receive an analysis of an ABC score from the ABC Expert.
Your job is to answer the user's question based ONLY on that analysis.

ABC Expert Analysis:
{analysis}

Task:
{task_prompt}

Important:
- Output ONLY the option index (0,1,2,3,...)
- Do NOT output explanations
- Do NOT repeat the analysis
- Your answer must match one of the given options.
"""

def abc_expert_agent(input_abc):
    prompt = abc_expert_prompt(input_abc)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


def evaluator_agent(analysis, full_prompt, num_options):
    prompt = evaluator_prompt(analysis, full_prompt)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content
    return extract_option_index(raw, num_options)



predictions = []
taus = []
correct = 0

for i, row in df.iterrows():
    input_abc = row["score"]
    prompt = row["prompt"] 

    try:
        analysis = abc_expert_agent(input_abc)
        # print(analysis)
        pred = evaluator_agent(analysis, prompt, 4)
        
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


output_path = "metadata_QA_agent_gemma_results.csv"
df.to_csv(output_path, index=False)

print(f"Saved predictions to {output_path}")