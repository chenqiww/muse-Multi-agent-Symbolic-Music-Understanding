import pandas as pd
from pathlib import Path
# path of data
data_dir = Path("data")    


# ==============================
# 1. Error Detection
# ==============================


error_detection_csv = data_dir / "error_detection.csv"
ed_df = pd.read_csv(error_detection_csv)

def parse_error_list(err_string):
    return [e.strip() for e in err_string.split(",")]
    
ed_df["error_list"] = ed_df["error"].apply(parse_error_list)

def build_prompt_ed(row):
    input_content = row["input"]
    task = row["task_description"]
    options = row["error_list"]

    # build the options text block
    options_text = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)])

    prompt = f"""Input:
{input_content}

Task:
{task}

Options:
{options_text}
"""
    return prompt

ed_df["prompt"] = ed_df.apply(build_prompt_ed, axis=1)


ed_clean_df = ed_df.drop(columns = ["title","input","choices","target","task_description","error", "error_list"])
ed_clean_df = ed_clean_df.rename(columns={"target_index": "solution"})

output_path = data_dir / "Error_Detection_cleaned.csv"
ed_clean_df.to_csv(output_path, index=False)




# ==============================
# 2. Metadata_QA
# ==============================


metadata_qa = data_dir / "Metadata_QA.csv"
qa_df = pd.read_csv(metadata_qa)


import ast

def build_prompt_qa(row):
    input_content = row["score"]
    task = row["task_description"]

    # Convert string list "['Dm','D','G','E']" → real Python list
    choices = ast.literal_eval(row["choices"])

    # Format as:
    # 0. A 1. B
    # 2. C 3. D
    options_text = (
        f"0. {choices[0]}   1. {choices[1]}\n"
        f"2. {choices[2]}   3. {choices[3]}"
    )

    prompt = f"""Input:
{input_content}

Task:
{task}

Options:
{options_text}

Answer:"""

    return prompt


qa_df["prompt"] = qa_df.apply(build_prompt_qa, axis=1)


qa_clean_df = qa_df.drop(columns = ["title", "choices", "target","task_description"])
qa_clean_df = qa_clean_df.rename(columns={"target_index": "solution"})
output_path = data_dir / "Metadata_QA_cleaned.csv"
qa_clean_df.to_csv(output_path, index=False)




# ==============================
# 2. Emotion_Recognition 
# ==============================


metadata_er = data_dir / "Emotion_Recognition.csv"
er_df = pd.read_csv(metadata_er)


import ast

def build_prompt_er(row):
    input_content = row["score"]
    task = row["task_description"]


    prompt = f"""Input:
{input_content}

Task:
{task}

Options:
0. Q1      1. Q2
2. Q3      3. Q4

Answer:"""

    return prompt


er_df["prompt"] = er_df.apply(build_prompt_er, axis=1)


er_clean_df = er_df.drop(columns = ["title","score", "choices", "target","task_description"])
er_clean_df = er_clean_df.rename(columns={"target_index": "solution"})
output_path = data_dir / "Emotion_Recognition_cleaned.csv"
er_clean_df.to_csv(output_path, index=False)



# barsequence = data_dir / "Bar_Sequencing.csv"
# bs_df = pd.read_csv(barsequence)


# def build_prompt_bs(row):
#     input_content = row["input"]
#     task = row["task_description"]


#     prompt = f"""Input:
# {input_content}

# Task:
# {task}

# Options:
# there are {len(row["choices"])} bars: 
# {row["choices"]}

# Answer:"""
#     return prompt

# bs_df["prompt"] = bs_df.apply(build_prompt_bs, axis=1)


# bs_clean_df = bs_df.drop(columns = ["title","input", "choices", "target_index","task_description"])
# bs_clean_df = bs_clean_df.rename(columns={"target": "solution"})
# output_path = data_dir / "Bar_Sequencing_cleaned.csv"
# bs_clean_df.to_csv(output_path, index=False)




import pandas as pd
import re

barsequence = data_dir / "Bar_Sequencing.csv"
bs_df = pd.read_csv(barsequence)


def parse_choices(choices_str: str):
    """
    把像 ['""G""d2B', '""D""A2e', '""D""A3/2B/2c', '""G""B2G']
    这种字符串解析成一个 list:
    ['""G""d2B', '""D""A2e', '""D""A3/2B/2c', '""G""B2G']
    不用 literal_eval，避免因为引号/反斜杠炸掉。
    """
    s = choices_str.strip()

    # 先尝试用单引号括起来的 item 解析
    items = re.findall(r"'(.*?)'", s)
    if items:
        return [x.strip() for x in items]

    # 如果没有单引号，就尝试双引号
    items = re.findall(r'"(.*?)"', s)
    if items:
        return [x.strip() for x in items]

    # 实在不行，就当成一个整体（极端 fallback）
    return [s]


def build_prompt_bs(row):
    input_content = row["input"]
    task = row["task_description"]

    # 解析 choices
    choices_list = parse_choices(row["choices"])

    # 编号形式：0., 1., 2., ...
    options_text = "\n".join(
        f"{i}. {bar}"
        for i, bar in enumerate(choices_list)
    )

    # 这里只负责组织 input/task/options，格式要求你之后再 concat
    prompt = f"""
Task:
{task}
Input:
{input_content}


Options:
{options_text}

Answer:"""

    return prompt


# 构建 prompt 列
bs_df["prompt"] = bs_df.apply(build_prompt_bs, axis=1)

# 只保留 prompt + solution
bs_clean_df = bs_df[["prompt", "target"]].rename(columns={"target": "solution"})

# 输出 clean CSV
output_path = data_dir / "Bar_Sequencing_cleaned.csv"
bs_clean_df.to_csv(output_path, index=False)

print("Clean file saved →", output_path)
