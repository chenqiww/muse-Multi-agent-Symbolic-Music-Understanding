from openai import OpenAI, APIConnectionError, APITimeoutError
from inference_auth_token import get_access_token
import pandas as pd
import re
import sys
from collections import Counter

model_name = "google/gemma-3-27b-it"
#model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"

client = OpenAI(
    api_key=get_access_token(),
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)

# Emotion recognition system components
category_text = """Categories:
0: Q1 (happy   - high valence, high arousal)
1: Q2 (angry   - low  valence, high arousal)
2: Q3 (sad     - low  valence, low  arousal)
3: Q4 (relaxed - high valence, low  arousal)
"""

def call_llm(prompt, max_tokens=128, temperature=0.5):
    """Helper function to call LLM."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n"] if max_tokens <= 8 else None
        )
        content = response.choices[0].message.content
        if content is None:
            print(f"Warning: LLM returned None content")
            return ""
        return content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

def extract_abc_from_prompt(user_prompt):
    """Extract ABC score from user prompt."""
    # Try to extract ABC from triple backticks
    abc_match = re.search(r"```(.*?)```", user_prompt, re.S)
    if abc_match:
        return abc_match.group(1).strip()
    
    # Try "Input:" pattern - extract everything from "Input:" to "Task:" or "Options:"
    m1 = re.search(r"Input:\s*\n(.*?)(?=\n+Task:|\n+Options:)", user_prompt, re.S)
    if m1:
        result = m1.group(1).strip()
        if result:
            return result
    
    # Try "Score:" pattern
    m2 = re.search(r"Score:\s*\n(.*?)(?=\n+Task:|\n+Options:)", user_prompt, re.S)
    if m2:
        result = m2.group(1).strip()
        if result:
            return result
    
    # Fallback: try to find ABC notation patterns (lines starting with X:, K:, M:, etc.)
    lines = user_prompt.split('\n')
    abc_lines = []
    in_abc = False
    for line in lines:
        stripped = line.strip()
        if not stripped and not in_abc:
            continue
        # ABC notation typically starts with X:, K:, M:, L:, R:, or contains note patterns
        if re.match(r'^[XKMLR]:', stripped) or (stripped and re.search(r'[A-Ga-g][#b]?\d+', stripped)):
            abc_lines.append(stripped)
            in_abc = True
        elif in_abc:
            if stripped.startswith(('Task:', 'Options:', 'Answer:')):
                break
            elif stripped and not any(keyword in stripped for keyword in ['Input:', 'Task:', 'Options:', 'Answer:']):
                abc_lines.append(stripped)
            elif not stripped:
                if abc_lines:
                    abc_lines.append(stripped)
    
    if abc_lines:
        result = '\n'.join(abc_lines).strip()
        if result:
            return result
    
    # Last fallback: return the whole prompt if no pattern matches
    return user_prompt.strip()

def validate_input_simple(user_prompt):
    """
    Simplified input validation for emotion classification only.
    Returns: (is_valid, error_message, verified_abc)
    """
    # Step 1: Try to extract ABC score using script
    extracted_abc = extract_abc_from_prompt(user_prompt)
    has_abc = extracted_abc and len(extracted_abc) > 10 and any(c in extracted_abc for c in ['X:', 'K:', 'M:', 'L:'])
    
    if not has_abc:
        # Call LLM to try extracting ABC
        validation_prompt = f"""You are an input validator for an emotion classification system.

Your task is to check if the user input contains an ABC notation score.

User input:
{user_prompt}

ABC notation typically:
- Starts with headers like X:, K:, M:, L:, R:
- Contains musical notes (A-G with optional sharps/flats and octaves)
- May be wrapped in code blocks (```) or after "Input:" or "Score:"

Please analyze the user input and:
1. If you find ABC notation, extract it completely and respond with:
   EXTRACTED_ABC:
   [the complete ABC score here]

2. If you confirm there is NO ABC notation, respond with:
   NO_ABC_SCORE

Your response:"""
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0,
                max_tokens=300
            )
            validation_result = response.choices[0].message.content.strip()
            
            if "NO_ABC_SCORE" in validation_result.upper():
                return (False, "No ABC score detected in your input. Please include an ABC notation score and try again.", None)
            elif "EXTRACTED_ABC:" in validation_result:
                abc_match = re.search(r"EXTRACTED_ABC:\s*\n(.*?)(?=\n\n|\n[A-Z_]+:|$)", validation_result, re.S)
                if abc_match:
                    verified_abc = abc_match.group(1).strip()
                    extracted_abc = verified_abc
                    has_abc = True
                else:
                    return (False, "Could not extract ABC score. Please format your input clearly with ABC notation.", None)
        except Exception as e:
            return (False, f"Input validation error: {str(e)}", None)
    
    # Step 2: Check if user asked a question
    question_keywords = ['what', 'which', 'how', 'classify', 'emotion', 'mood', 'feeling', 'q1', 'q2', 'q3', 'q4', 'task:', 'question']
    has_question = any(keyword in user_prompt.lower() for keyword in question_keywords)
    
    if not has_question:
        return (False, "No question detected in your input. Please ask a question about the emotion classification of the music score.", extracted_abc if has_abc else None)
    
    return (True, None, extracted_abc if has_abc else None)

def build_arousal_classifier_prompt(abc_score):
    """Build prompt for arousal classifier (HIGH or LOW)."""
    prompt = f"""Arousal classification:
- HIGH arousal: energetic, intense, driving
- LOW arousal: calm, peaceful, relaxed

Output format:
AROUSAL: <HIGH or LOW>
REASON: <brief explanation>

{abc_score}

AROUSAL:"""
    return prompt

def build_valence_classifier_prompt(abc_score):
    """Build prompt for valence classifier (HIGH or LOW)."""
    prompt = f"""Valence classification:
- HIGH valence: pleasant, bright, joyful
- LOW valence: unpleasant, dark, sad

Output format:
VALENCE: <HIGH or LOW>
REASON: <brief explanation>

{abc_score}

VALENCE:"""
    return prompt

def build_emotion_combiner_prompt(abc_score, arousal_result, valence_result):
    """Build prompt for combining arousal and valence into final emotion category."""
    prompt = f"""{category_text}

Mapping:
- High Valence + High Arousal → Label 0
- Low Valence + High Arousal → Label 1
- Low Valence + Low Arousal → Label 2
- High Valence + Low Arousal → Label 3

{abc_score}

Arousal: {arousal_result}
Valence: {valence_result}

Answer:"""
    return prompt

def extract_reason(text: str) -> str:
    """Extract reason from analyst answer."""
    if not text:
        return ""
    lines = str(text).splitlines()
    for line in lines:
        if line.strip().upper().startswith("REASON:"):
            return line.split(":", 1)[1].strip()
    return str(text).strip()

def extract_arousal(text: str) -> str:
    """Extract arousal level (HIGH or LOW) from text."""
    if not text:
        return ""
    text_upper = text.upper()
    if "AROUSAL:" in text_upper:
        arousal_line = [l for l in text.splitlines() if 'AROUSAL:' in l.upper()]
        if arousal_line:
            if "HIGH" in arousal_line[0].upper():
                return "HIGH"
            elif "LOW" in arousal_line[0].upper():
                return "LOW"
    # Fallback: search for HIGH or LOW
    if "HIGH" in text_upper and "AROUSAL" in text_upper:
        return "HIGH"
    elif "LOW" in text_upper and "AROUSAL" in text_upper:
        return "LOW"
    return ""

def extract_valence(text: str) -> str:
    """Extract valence level (HIGH or LOW) from text."""
    if not text:
        return ""
    text_upper = text.upper()
    if "VALENCE:" in text_upper:
        valence_line = [l for l in text.splitlines() if 'VALENCE:' in l.upper()]
        if valence_line:
            if "HIGH" in valence_line[0].upper():
                return "HIGH"
            elif "LOW" in valence_line[0].upper():
                return "LOW"
    # Fallback: search for HIGH or LOW
    if "HIGH" in text_upper and "VALENCE" in text_upper:
        return "HIGH"
    elif "LOW" in text_upper and "VALENCE" in text_upper:
        return "LOW"
    return ""

def classify_arousal(abc_score, num_analysts=3):
    """
    Classify arousal level (HIGH or LOW) using multiple analysts.
    Returns: (arousal_level, arousal_reason)
    """
    arousal_predictions = []
    arousal_reasons = []
    
    print("  Classifying arousal level...")
    for k in range(num_analysts):
        prompt = build_arousal_classifier_prompt(abc_score)
        answer = call_llm(prompt, max_tokens=128, temperature=0.4)
        
        arousal = extract_arousal(answer)
        reason = extract_reason(answer)
        
        arousal_predictions.append(arousal)
        arousal_reasons.append(reason)
        print(f"    Arousal Analyst {k+1}: {arousal}")
    
    # Majority vote for arousal
    valid_arousals = [a for a in arousal_predictions if a in ["HIGH", "LOW"]]
    if valid_arousals:
        final_arousal = Counter(valid_arousals).most_common(1)[0][0]
    else:
        final_arousal = ""
    
    # Combine reasons
    combined_reason = " | ".join([r for r in arousal_reasons if r])
    
    return final_arousal, combined_reason

def classify_valence(abc_score, num_analysts=3):
    """
    Classify valence level (HIGH or LOW) using multiple analysts.
    Returns: (valence_level, valence_reason)
    """
    valence_predictions = []
    valence_reasons = []
    
    print("  Classifying valence level...")
    for k in range(num_analysts):
        prompt = build_valence_classifier_prompt(abc_score)
        answer = call_llm(prompt, max_tokens=128, temperature=0.4)
        
        valence = extract_valence(answer)
        reason = extract_reason(answer)
        
        valence_predictions.append(valence)
        valence_reasons.append(reason)
        print(f"    Valence Analyst {k+1}: {valence}")
    
    # Majority vote for valence
    valid_valences = [v for v in valence_predictions if v in ["HIGH", "LOW"]]
    if valid_valences:
        final_valence = Counter(valid_valences).most_common(1)[0][0]
    else:
        final_valence = ""
    
    # Combine reasons
    combined_reason = " | ".join([r for r in valence_reasons if r])
    
    return final_valence, combined_reason

# Single-call versions for testing (no voting)
def classify_arousal_single(abc_score):
    """
    Classify arousal level (HIGH or LOW) using single call.
    Returns: (arousal_level, arousal_reason)
    """
    print("  Classifying arousal level (single call)...")
    prompt = build_arousal_classifier_prompt(abc_score)
    answer = call_llm(prompt, max_tokens=128, temperature=0.4)
    
    arousal = extract_arousal(answer)
    reason = extract_reason(answer)
    
    print(f"    Arousal: {arousal}")
    
    return arousal, reason

def classify_valence_single(abc_score):
    """
    Classify valence level (HIGH or LOW) using single call.
    Returns: (valence_level, valence_reason)
    """
    print("  Classifying valence level (single call)...")
    prompt = build_valence_classifier_prompt(abc_score)
    answer = call_llm(prompt, max_tokens=128, temperature=0.4)
    
    valence = extract_valence(answer)
    reason = extract_reason(answer)
    
    print(f"    Valence: {valence}")
    
    return valence, reason

def emotion_classification_system(user_prompt):
    """
    Emotion recognition system using arousal-valence approach.
    Step 1: Classify arousal (HIGH/LOW) using multiple analysts
    Step 2: Classify valence (HIGH/LOW) using multiple analysts
    Step 3: Combine arousal and valence to get final emotion category (Q1-Q4)
    
    Returns: (response_string, arousal_level, valence_level, final_label)
    """
    # Extract ABC score from prompt
    abc_score = extract_abc_from_prompt(user_prompt)
    
    # Debug: check if extraction worked
    if not abc_score or len(abc_score) < 10 or not any(c in abc_score for c in ['X:', 'K:', 'M:', 'L:']):
        error_msg = f"Error: Could not extract ABC score from the prompt. Extracted content: {abc_score[:100]}..." if ("Input:" in user_prompt or "Score:" in user_prompt) else "Error: Could not extract ABC score. Please ensure the prompt contains ABC notation after 'Input:' or 'Score:'."
        return (error_msg, "", "", "")
    
    num_analysts = 3
    
    # Step 1: Classify arousal (HIGH or LOW)
    arousal_level, arousal_reason = classify_arousal(abc_score, num_analysts)
    
    # Step 2: Classify valence (HIGH or LOW)
    valence_level, valence_reason = classify_valence(abc_score, num_analysts)
    
    # Step 3: Combine arousal and valence to get final emotion category
    print("  Combining arousal and valence...")
    arousal_result = f"Arousal: {arousal_level}\nReason: {arousal_reason}"
    valence_result = f"Valence: {valence_level}\nReason: {valence_reason}"
    
    combiner_prompt = build_emotion_combiner_prompt(abc_score, arousal_result, valence_result)
    combiner_answer = call_llm(combiner_prompt, max_tokens=4, temperature=0.0)
    
    # Extract final label
    final_label_match = re.search(r'\b([0-3])\b', combiner_answer)
    if final_label_match:
        final_label = final_label_match.group(1)
    else:
        # Fallback: map arousal+valence directly
        if arousal_level == "HIGH" and valence_level == "HIGH":
            final_label = "0"  # Q1
        elif arousal_level == "HIGH" and valence_level == "LOW":
            final_label = "1"  # Q2
        elif arousal_level == "LOW" and valence_level == "LOW":
            final_label = "2"  # Q3
        elif arousal_level == "LOW" and valence_level == "HIGH":
            final_label = "3"  # Q4
        else:
            final_label = ""
    
    # Map label to emotion name
    emotion_map = {"0": "Q1 (happy)", "1": "Q2 (angry)", "2": "Q3 (sad)", "3": "Q4 (relaxed)"}
    emotion_name = emotion_map.get(final_label, "Unknown")
    
    # Build response with explanation
    response = f"Emotion Classification: {emotion_name} (Label: {final_label})\n\n"
    response += f"Arousal Classification: {arousal_level}\n"
    response += f"  Reasoning: {arousal_reason}\n\n"
    response += f"Valence Classification: {valence_level}\n"
    response += f"  Reasoning: {valence_reason}\n\n"
    response += f"Combined Result: {arousal_level} arousal + {valence_level} valence → {emotion_name}\n"
    
    return (response, arousal_level, valence_level, final_label)

def emotion_classification_system_single(user_prompt):
    """
    Emotion recognition system using arousal-valence approach (single call version).
    Step 1: Classify arousal (HIGH/LOW) using single call
    Step 2: Classify valence (HIGH/LOW) using single call
    Step 3: Combine arousal and valence to get final emotion category (Q1-Q4)
    
    Returns: (response_string, arousal_level, valence_level, final_label)
    """
    # Extract ABC score from prompt
    abc_score = extract_abc_from_prompt(user_prompt)
    
    # Debug: check if extraction worked
    if not abc_score or len(abc_score) < 10 or not any(c in abc_score for c in ['X:', 'K:', 'M:', 'L:']):
        error_msg = f"Error: Could not extract ABC score from the prompt. Extracted content: {abc_score[:100]}..." if ("Input:" in user_prompt or "Score:" in user_prompt) else "Error: Could not extract ABC score. Please ensure the prompt contains ABC notation after 'Input:' or 'Score:'."
        return (error_msg, "", "", "")
    
    # Step 1: Classify arousal (HIGH or LOW) - single call
    arousal_level, arousal_reason = classify_arousal_single(abc_score)
    
    # Step 2: Classify valence (HIGH or LOW) - single call
    valence_level, valence_reason = classify_valence_single(abc_score)
    
    # Step 3: Combine arousal and valence to get final emotion category
    print("  Combining arousal and valence...")
    arousal_result = f"Arousal: {arousal_level}\nReason: {arousal_reason}"
    valence_result = f"Valence: {valence_level}\nReason: {valence_reason}"
    
    combiner_prompt = build_emotion_combiner_prompt(abc_score, arousal_result, valence_result)
    combiner_answer = call_llm(combiner_prompt, max_tokens=4, temperature=0.0)
    
    # Extract final label
    final_label_match = re.search(r'\b([0-3])\b', combiner_answer)
    if final_label_match:
        final_label = final_label_match.group(1)
    else:
        # Fallback: map arousal+valence directly
        if arousal_level == "HIGH" and valence_level == "HIGH":
            final_label = "0"  # Q1
        elif arousal_level == "HIGH" and valence_level == "LOW":
            final_label = "1"  # Q2
        elif arousal_level == "LOW" and valence_level == "LOW":
            final_label = "2"  # Q3
        elif arousal_level == "LOW" and valence_level == "HIGH":
            final_label = "3"  # Q4
        else:
            final_label = ""
    
    # Map label to emotion name
    emotion_map = {"0": "Q1 (happy)", "1": "Q2 (angry)", "2": "Q3 (sad)", "3": "Q4 (relaxed)"}
    emotion_name = emotion_map.get(final_label, "Unknown")
    
    # Build response with explanation
    response = f"Emotion Classification: {emotion_name} (Label: {final_label})\n\n"
    response += f"Arousal Classification: {arousal_level}\n"
    response += f"  Reasoning: {arousal_reason}\n\n"
    response += f"Valence Classification: {valence_level}\n"
    response += f"  Reasoning: {valence_reason}\n\n"
    response += f"Combined Result: {arousal_level} arousal + {valence_level} valence → {emotion_name}\n"
    
    return (response, arousal_level, valence_level, final_label)

def run_emotion_system(user_prompt, use_single=False):
    """
    Main entry point for emotion classification system.
    Validates input, then runs emotion classification.
    
    Args:
        user_prompt: Input prompt with ABC score
        use_single: If True, use single-call version; if False, use 3-vote version
    
    Returns: (response_string, arousal_level, valence_level, final_label)
    """
    # Step 0: Validate input (check ABC score and question)
    print("Validating input...")
    is_valid, error_message, verified_abc = validate_input_simple(user_prompt)
    
    if not is_valid:
        error_response = f"❌ Input Validation Error:\n{error_message}\n\nPlease correct your input and try again."
        return (error_response, "", "", "")
    
    if verified_abc:
        print(f"✓ ABC score validated ({len(verified_abc)} characters)")
    
    # Step 1: Run emotion classification
    print("Running emotion classification...")
    if use_single:
        return emotion_classification_system_single(user_prompt)
    else:
        return emotion_classification_system(user_prompt)


# Main program entry point
if __name__ == "__main__":
    import sys
    
    # Check if running in batch mode (with CSV file) or interactive mode
    if len(sys.argv) > 1:
        # Batch mode: process CSV file
        csv_path = sys.argv[1]
        print(f"Processing CSV file: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Check if this is the rough4q format (has 'data' and 'label' columns)
        is_rough4q_format = "data" in df.columns and "label" in df.columns
        
        if is_rough4q_format:
            print("Detected rough4q_full_raw.csv format")
            # Build prompt from data column (ABC score)
            def build_prompt_from_data(row):
                abc_data = str(row.get("data", "")).strip()
                if not abc_data:
                    return ""
                # Build standard prompt format
                prompt = f"""Input:
{abc_data}

Task:
Choose the most probable emotional label of the provided score. Label Q1 refers to happy (high valence high arousal), Q2 refers to angry (low valence high arousal), Q3 refers to sad (low valence low arousal) and Q4 refers to relaxed (high valence low arousal).

Options:
0. Q1      1. Q2
2. Q3      3. Q4

Answer:"""
                return prompt
            
            df["prompt"] = df.apply(build_prompt_from_data, axis=1)
            # Use 'label' column as ground truth (instead of 'solution')
            df["solution"] = df["label"].astype(str)
            print(f"Total samples: {len(df)}")
            
            # Optional: filter by split (train/test)
            if len(sys.argv) > 3 and sys.argv[3] in ["train", "test", "val"]:
                split_filter = sys.argv[3]
                df = df[df["split"] == split_filter]
                print(f"Filtered to '{split_filter}' split: {len(df)} samples")
        else:
            # Original format (Emotion_Recognition_cleaned.csv)
            if "prompt" not in df.columns:
                print("Error: CSV file must have a 'prompt' column (or 'data' column for rough4q format)")
                sys.exit(1)
            print("Using original format (prompt column)")
        
        # Check for --single flag (use single-call version instead of 3-vote version)
        use_single = "--single" in sys.argv
        if use_single:
            print("Using single-call version (no voting)")
        
        # Optional: limit number of samples for testing (random sampling)
        # Handle both --n=20 and --single cases
        n_samples = None
        for arg in sys.argv[2:]:
            if arg.startswith('--') and arg[2:].isdigit():
                n_samples = int(arg[2:])
                break
        
        if n_samples:
            n_samples = min(n_samples, len(df))  # Ensure n doesn't exceed total samples
            df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
            print(f"Randomly sampling {n_samples} samples for testing")
        
        results = []
        pred_arousals = []
        pred_valences = []
        pred_labels = []
        correct = 0
        correct_arousal = 0
        correct_valence = 0
        
        # Helper function to get ground truth arousal/valence from label
        def get_gt_arousal_valence(label_str):
            """Get ground truth arousal and valence from label (0-3).
            Returns: (arousal, valence)
            """
            label_map = {
                "0": ("HIGH", "HIGH"),  # Q1: high valence, high arousal → arousal=HIGH, valence=HIGH
                "1": ("HIGH", "LOW"),   # Q2: low valence, high arousal → arousal=HIGH, valence=LOW
                "2": ("LOW", "LOW"),    # Q3: low valence, low arousal → arousal=LOW, valence=LOW
                "3": ("LOW", "HIGH")    # Q4: high valence, low arousal → arousal=LOW, valence=HIGH
            }
            return label_map.get(str(label_str), ("", ""))
        
        for i, row in df.iterrows():
            print(f"\n{'='*60}")
            print(f"Processing sample {i+1}/{len(df)}...")
            user_prompt = row["prompt"]
            
            try:
                answer, pred_arousal, pred_valence, pred_label = run_emotion_system(user_prompt, use_single=use_single)
                results.append(answer)
                pred_arousals.append(pred_arousal)
                pred_valences.append(pred_valence)
                pred_labels.append(pred_label)
                
                # Compare with ground truth (support both 'solution' and 'label' columns)
                gt_label = str(row.get("solution", row.get("label", "")))
                gt_arousal, gt_valence = get_gt_arousal_valence(gt_label)
                
                # Check overall label accuracy
                if pred_label == gt_label:
                    correct += 1
                    status = "✓ CORRECT"
                else:
                    status = "✗ WRONG"
                
                # Check arousal accuracy
                if pred_arousal == gt_arousal and gt_arousal:
                    correct_arousal += 1
                
                # Check valence accuracy
                if pred_valence == gt_valence and gt_valence:
                    correct_valence += 1
                
                label_name = row.get("label_name", f"Q{int(gt_label)+1 if gt_label.isdigit() else '?'}")
                print(f"\n{status} | GT={gt_label} ({label_name}) | Pred={pred_label}")
                print(f"  Arousal: GT={gt_arousal} | Pred={pred_arousal} {'✓' if pred_arousal == gt_arousal and gt_arousal else '✗'}")
                print(f"  Valence: GT={gt_valence} | Pred={pred_valence} {'✓' if pred_valence == gt_valence and gt_valence else '✗'}")
                print(f"Answer preview: {answer[:150]}...")
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                results.append(f"Error: {str(e)}")
                pred_arousals.append("")
                pred_valences.append("")
                pred_labels.append("")
        
        # Save results (create a copy to avoid modifying original df)
        results_df = df.copy()
        results_df["agent_answer"] = results
        results_df["predicted_label"] = pred_labels
        results_df["predicted_arousal"] = pred_arousals
        results_df["predicted_valence"] = pred_valences
        
        # Add ground truth arousal/valence for reference
        gt_arousals = []
        gt_valences = []
        for _, row in df.iterrows():
            gt_label = str(row.get("solution", row.get("label", "")))
            gt_arousal, gt_valence = get_gt_arousal_valence(gt_label)
            gt_arousals.append(gt_arousal)
            gt_valences.append(gt_valence)
        results_df["ground_truth_arousal"] = gt_arousals
        results_df["ground_truth_valence"] = gt_valences
        
        # Generate output filename based on mode
        suffix = "_single" if use_single else ""
        output_path = csv_path.replace(".csv", f"_emotion_test_results{suffix}.csv")
        results_df.to_csv(output_path, index=False)
        
        # Calculate accuracies
        total = len(df)
        accuracy = correct / total if total > 0 else 0
        arousal_accuracy = correct_arousal / total if total > 0 else 0
        valence_accuracy = correct_valence / total if total > 0 else 0
        
        print(f"\n{'='*60}")
        mode_str = "Single-call" if use_single else "3-vote"
        print(f"Mode: {mode_str}")
        print(f"Results saved to: {output_path}")
        print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.4f}")
        print(f"Arousal Accuracy: {correct_arousal}/{total} = {arousal_accuracy:.4f}")
        print(f"Valence Accuracy: {correct_valence}/{total} = {valence_accuracy:.4f}")
        
    else:
        # Interactive mode
        use_single = "--single" in sys.argv
        print("=" * 60)
        print("Emotion Classification Test System")
        if use_single:
            print("Mode: Single-call (no voting)")
        else:
            print("Mode: 3-vote (default)")
        print("=" * 60)
        print("\nEnter your question about emotion classification.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                answer, _, _, _ = run_emotion_system(user_input, use_single=use_single)
                print(f"\nSystem:\n{answer}\n")
            except Exception as e:
                print(f"\nError: {e}\n")

