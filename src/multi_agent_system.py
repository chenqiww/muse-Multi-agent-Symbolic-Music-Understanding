from openai import OpenAI, APIConnectionError, APITimeoutError
from inference_auth_token import get_access_token
import pandas as pd
import re
from collections import Counter

model_name = "google/gemma-3-27b-it"
#model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"

client = OpenAI(
    api_key=get_access_token(),
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)




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
- If the question asks for a specific option index (0, 1, 2, 3, etc.), output ONLY that number.
- If the question asks for a general answer, provide a clear and concise answer based on the analysis.
- Base your answer ONLY on the ABC Expert Analysis provided above.
"""

def abc_expert_agent(input_abc):
    prompt = abc_expert_prompt(input_abc)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


def extract_option_index(pred_raw, num_options=10):
    """Extract option index from model response."""
    if pred_raw is None:
        return ""
    
    text = pred_raw.strip()
    
    # 1) catch "**2." format
    m = re.search(r"\*\*(\d+)\s*[.)]", text)
    if m:
        idx = m.group(1)
        if int(idx) < num_options:
            return idx
    
    # 2) catch "**2**" format
    m = re.search(r"\*\*(\d+)\*\*", text)
    if m:
        idx = m.group(1)
        if int(idx) < num_options:
            return idx
    
    # 3) catch "2."、"2)"、"2 -" format
    m = re.search(r"\b(\d+)\s*[.)-]", text)
    if m:
        idx = m.group(1)
        if int(idx) < num_options:
            return idx
    
    # 4) catch simple number
    m = re.search(r"\b(\d+)\b", text)
    if m:
        idx = m.group(1)
        if int(idx) < num_options:
            return idx
    
    # 5) fallback (not found)
    return ""

def evaluator_agent(analysis, full_prompt, num_options, return_full=False):
    """
    Evaluator agent that answers questions based on ABC analysis.
    
    Args:
        analysis: ABC expert's analysis
        full_prompt: Original user prompt with task
        num_options: Number of options (for index extraction)
        return_full: If True, return full response; if False, return only option index
    
    Returns:
        Option index (string) if return_full=False, or full response if return_full=True
    """
    prompt = evaluator_prompt(analysis, full_prompt)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200 if return_full else 50
    )
    raw = response.choices[0].message.content.strip()
    
    if return_full:
        return raw
    else:
        # Extract option index
        return extract_option_index(raw, num_options)


def agent_B_abc_system(user_prompt):
    """
    ABC expert system: analyzes ABC score and answers questions about it.
    """
    # Extract ABC from user prompt using the unified extraction function
    abc_text = extract_abc_from_prompt(user_prompt)
    
    if not abc_text or len(abc_text) < 5:
        return "Error: Could not extract ABC score from the prompt."
    
    try:
        # Step 1: Expert analysis of ABC score
        analysis = abc_expert_agent(abc_text)
        
        if not analysis:
            return "Error: Failed to analyze ABC score."
        
        # Step 2: Determine number of options from the prompt
        # Check if prompt contains options (e.g., "0. A  1. B  2. C  3. D")
        options_match = re.findall(r'(\d+)\.\s*[^\d\n]+', user_prompt)
        num_options = len(options_match) if options_match else 4  # Default to 4
        
        # Step 3: Evaluator answers user question based on analysis
        # Get full response for better context
        full_answer = evaluator_agent(analysis, user_prompt, num_options, return_full=True)
        
        # Extract option index from full answer
        option_index = extract_option_index(full_answer, num_options)
        
        # If we found an option index and the full answer is just a number, return the full answer
        # Otherwise, return the full answer which should contain explanation
        if option_index and len(full_answer.strip()) <= 3:
            # Just a number, return it
            return option_index
        elif option_index and option_index in full_answer:
            # Full answer contains the option, return it
            return full_answer
        elif option_index:
            # We have an option but full answer doesn't contain it clearly
            return f"Answer: {option_index}\n\n{full_answer}"
        else:
            # No clear option index, return full answer
            return full_answer if full_answer else "Error: Could not determine answer."
        
    except Exception as e:
        return f"Error in ABC system: {str(e)}"

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
        return response.choices[0].message.content.strip()
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
    # This handles the format: Input:\nABC score\n\nTask:...
    # Match "Input:" followed by newline, then capture until "Task:" or "Options:"
    # Handle both single and double newlines before Task/Options
    m1 = re.search(r"Input:\s*\n(.*?)(?=\n+Task:|\n+Options:)", user_prompt, re.S)
    if m1:
        result = m1.group(1).strip()
        if result:  # Make sure we got something
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
        # Skip empty lines at the start
        if not stripped and not in_abc:
            continue
        # ABC notation typically starts with X:, K:, M:, L:, R:, or contains note patterns
        if re.match(r'^[XKMLR]:', stripped) or (stripped and re.search(r'[A-Ga-g][#b]?\d+', stripped)):
            abc_lines.append(stripped)
            in_abc = True
        elif in_abc:
            # Stop if we hit Task, Options, or Answer
            if stripped.startswith(('Task:', 'Options:', 'Answer:')):
                break
            # Continue if it's still part of ABC (notes, bars, etc.)
            elif stripped and not any(keyword in stripped for keyword in ['Input:', 'Task:', 'Options:', 'Answer:']):
                abc_lines.append(stripped)
            elif not stripped:
                # Empty line might be separator, but continue if we have content
                if abc_lines:
                    abc_lines.append(stripped)
    
    if abc_lines:
        result = '\n'.join(abc_lines).strip()
        if result:
            return result
    
    # Last fallback: return the whole prompt if no pattern matches
    return user_prompt.strip()

# Arousal-Valence based emotion classification functions
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
    
    for k in range(num_analysts):
        prompt = build_arousal_classifier_prompt(abc_score)
        answer = call_llm(prompt, max_tokens=128, temperature=0.4)
        
        arousal = extract_arousal(answer)
        reason = extract_reason(answer)
        
        arousal_predictions.append(arousal)
        arousal_reasons.append(reason)
    
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
    
    for k in range(num_analysts):
        prompt = build_valence_classifier_prompt(abc_score)
        answer = call_llm(prompt, max_tokens=128, temperature=0.4)
        
        valence = extract_valence(answer)
        reason = extract_reason(answer)
        
        valence_predictions.append(valence)
        valence_reasons.append(reason)
    
    # Majority vote for valence
    valid_valences = [v for v in valence_predictions if v in ["HIGH", "LOW"]]
    if valid_valences:
        final_valence = Counter(valid_valences).most_common(1)[0][0]
    else:
        final_valence = ""
    
    # Combine reasons
    combined_reason = " | ".join([r for r in valence_reasons if r])
    
    return final_valence, combined_reason

def agent_C_emotion_system(user_prompt):
    """
    Emotion recognition system using arousal-valence approach.
    Step 1: Classify arousal (HIGH/LOW) using multiple analysts
    Step 2: Classify valence (HIGH/LOW) using multiple analysts
    Step 3: Combine arousal and valence to get final emotion category (Q1-Q4)
    """
    # Extract ABC score from prompt
    abc_score = extract_abc_from_prompt(user_prompt)
    
    # Debug: check if extraction worked
    if not abc_score or len(abc_score) < 10 or not any(c in abc_score for c in ['X:', 'K:', 'M:', 'L:']):
        # Try to provide helpful error message
        if "Input:" in user_prompt or "Score:" in user_prompt:
            return f"Error: Could not extract ABC score from the prompt. Extracted content: {abc_score[:100]}..."
        else:
            return "Error: Could not extract ABC score. Please ensure the prompt contains ABC notation after 'Input:' or 'Score:'."
    
    num_analysts = 3
    
    # Step 1: Classify arousal (HIGH or LOW)
    arousal_level, arousal_reason = classify_arousal(abc_score, num_analysts)
    
    # Step 2: Classify valence (HIGH or LOW)
    valence_level, valence_reason = classify_valence(abc_score, num_analysts)
    
    # Step 3: Combine arousal and valence to get final emotion category
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
    
    return response


def input_validator_prompt(user_prompt, extracted_abc, has_abc):
    """Build prompt for input validation LLM."""
    if not has_abc:
        # No ABC score detected, ask LLM to try extracting
        return f"""You are an input validator for a music analysis system.

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

3. If the input is unclear or incomplete, respond with:
   UNCLEAR_INPUT

Your response:"""
    else:
        # ABC score extracted, verify it's complete and check for question
        return f"""You are an input validator for a music analysis system.

Your task is to:
1. Verify that the extracted ABC score is complete and correct
2. Check if the user has asked a question

User input:
{user_prompt}

Extracted ABC score:
{extracted_abc}

Please analyze and respond in one of these formats:

If the ABC score is complete and correct, AND the user has asked a question:
VALID_INPUT

If the ABC score is incomplete or missing parts:
INCOMPLETE_ABC:
[description of what's missing or what should be added]

If the user has NOT asked a question:
NO_QUESTION_DETECTED

If there are other issues:
ISSUE:
[description of the issue]

Your response:"""


def validate_input(user_prompt):
    """
    Validate user input before processing.
    Returns: (is_valid, error_message, verified_abc)
    - is_valid: True if input is valid, False otherwise
    - error_message: Error message if invalid, None if valid
    - verified_abc: Verified/extracted ABC score, or None
    """
    # Step 1: Try to extract ABC score using script
    extracted_abc = extract_abc_from_prompt(user_prompt)
    has_abc = extracted_abc and len(extracted_abc) > 10 and any(c in extracted_abc for c in ['X:', 'K:', 'M:', 'L:'])
    
    # Step 2: Call LLM for validation
    validation_prompt = input_validator_prompt(user_prompt, extracted_abc, has_abc)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": validation_prompt}],
            temperature=0,
            max_tokens=300
        )
        validation_result = response.choices[0].message.content.strip()
        
        # Parse validation result
        if not has_abc:
            # Case 1: No ABC detected initially
            if "NO_ABC_SCORE" in validation_result.upper():
                return (False, "No ABC score detected in your input. Please include an ABC notation score and try again.", None)
            elif "EXTRACTED_ABC:" in validation_result:
                # LLM found ABC score, extract it
                abc_match = re.search(r"EXTRACTED_ABC:\s*\n(.*?)(?=\n\n|\n[A-Z_]+:|$)", validation_result, re.S)
                if abc_match:
                    verified_abc = abc_match.group(1).strip()
                    # Continue to check for question
                    return validate_input_with_abc(user_prompt, verified_abc)
                else:
                    # Try to extract from the rest of the response
                    lines = validation_result.split("EXTRACTED_ABC:")[1].strip().split("\n")
                    verified_abc = "\n".join([l for l in lines if not l.strip().startswith(("NO_", "UNCLEAR", "VALID", "INCOMPLETE", "ISSUE"))]).strip()
                    if verified_abc:
                        return validate_input_with_abc(user_prompt, verified_abc)
                    else:
                        return (False, "Could not extract ABC score. Please format your input clearly with ABC notation.", None)
            elif "UNCLEAR_INPUT" in validation_result.upper():
                return (False, "Input is unclear. Please provide a clear ABC notation score and a question.", None)
            else:
                # LLM might have extracted ABC in a different format
                return (False, "No ABC score detected. Please include an ABC notation score and try again.", None)
        else:
            # Case 2: ABC extracted, verify completeness and check for question
            return validate_input_with_abc(user_prompt, extracted_abc, validation_result)
            
    except Exception as e:
        print(f"Error in input validation: {e}")
        # If validation fails but we have ABC, proceed with warning
        if has_abc:
            return (True, None, extracted_abc)
        else:
            return (False, f"Input validation error: {str(e)}", None)


def validate_input_with_abc(user_prompt, abc_score, validation_result=None):
    """Validate input when ABC score is present."""
    if validation_result is None:
        # Need to call LLM to validate
        validation_prompt = input_validator_prompt(user_prompt, abc_score, True)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0,
                max_tokens=200
            )
            validation_result = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in ABC validation: {e}")
            # Proceed with extracted ABC
            return (True, None, abc_score)
    
    # Parse validation result
    validation_upper = validation_result.upper()
    
    if "VALID_INPUT" in validation_upper:
        return (True, None, abc_score)
    elif "NO_QUESTION_DETECTED" in validation_upper:
        return (False, "No question detected in your input. Please ask a question about the music score and try again.", abc_score)
    elif "INCOMPLETE_ABC:" in validation_result:
        # Extract description of what's missing
        missing_info = validation_result.split("INCOMPLETE_ABC:")[1].strip()
        return (False, f"ABC score appears to be incomplete. {missing_info} Please provide the complete ABC notation.", abc_score)
    elif "ISSUE:" in validation_result:
        issue_info = validation_result.split("ISSUE:")[1].strip()
        return (False, f"Input issue detected: {issue_info}", abc_score)
    else:
        # If unclear response but we have ABC, proceed with warning
        print(f"Warning: Unclear validation response, proceeding with extracted ABC")
        return (True, None, abc_score)


def controller_prompt(user_prompt):
    return f"""
You are the Controller Agent.

Your job is to decide which specialized agents should be used based on the USER'S QUESTIONS, not the presence of ABC score.

IMPORTANT: The presence of an ABC score in the prompt does NOT mean there is an ABC question. Only respond "ABC" or "BOTH" if the user EXPLICITLY ASKS a question about music theory or ABC-related topics.

Rules:
- If the user asks about music theory, ABC notation, music score structure, bars, keys, time signature, meter, chord symbols, or any technical music analysis questions -> respond "ABC"
- If the user asks about emotion, emotional label, valence/arousal, mood detection, Q1/Q2/Q3/Q4, happy/angry/sad/relaxed -> respond "EMOTION"
- If both topics appear → respond "BOTH"
- If neither → respond "NONE"

Key indicators for EMOTION:
- Words: emotion, emotional, mood, valence, arousal, feeling
- Labels: Q1, Q2, Q3, Q4
- Emotions: happy, angry, sad, relaxed
- Questions about emotional content or mood

Key indicators for ABC (ONLY if user explicitly asks):
- Questions about music theory concepts
- Questions about key, signature, bars, measures, time signature, meter
- Questions about chords, structure, notation, or technical music analysis
- Any question that requires analyzing the musical structure or notation

CRITICAL: If the prompt contains an ABC score but the user ONLY asks about emotion, respond "EMOTION" (not "BOTH" or "ABC"). The ABC score is just the input data, not a question.

Return ONLY ONE WORD: ABC, EMOTION, BOTH, or NONE.

User prompt:
{user_prompt}
"""


def agent_A_controller(user_prompt):
    """
    LLM-based controller that decides which agents to use.
    """
    decision = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": controller_prompt(user_prompt)}],
        temperature=0
    ).choices[0].message.content.strip()

    # Normalize
    decision = decision.upper()
    
    # Ensure valid decision
    if decision not in ("ABC", "EMOTION", "BOTH", "NONE"):
        # Fallback: try to extract from response
        if "BOTH" in decision:
            decision = "BOTH"
        elif "EMOTION" in decision:
            decision = "EMOTION"
        elif "ABC" in decision:
            decision = "ABC"
        else:
            decision = "NONE"

    return decision


def split_tasks_for_agents(user_prompt):
    """
    When decision is BOTH, split the prompt into ABC-related and Emotion-related tasks.
    Uses LLM to intelligently extract relevant parts for each agent.
    """
    split_prompt = f"""
You are a task splitter. Given a user prompt that contains both ABC notation questions and emotion classification questions, split it into two separate tasks.

Original prompt:
{user_prompt}

Extract and format:
1. ABC Task: The part of the prompt related to ABC notation, music structure, keys, meters, bars, chords, etc.
2. Emotion Task: The part of the prompt related to emotion, mood, valence, arousal, Q1/Q2/Q3/Q4 classification, etc.

Format your response as:
ABC_TASK:
[the ABC-related task here, including the ABC score if present]

EMOTION_TASK:
[the emotion-related task here, including the ABC score if present]

If the original prompt contains an ABC score, include it in BOTH tasks.
"""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": split_prompt}],
            temperature=0,
            max_tokens=500
        )
        split_text = response.choices[0].message.content.strip()
        
        # Parse the response
        abc_task = None
        emotion_task = None
        
        if "ABC_TASK:" in split_text:
            parts = split_text.split("ABC_TASK:")
            if len(parts) > 1:
                remaining = parts[1]
                if "EMOTION_TASK:" in remaining:
                    abc_task = remaining.split("EMOTION_TASK:")[0].strip()
                    emotion_task = remaining.split("EMOTION_TASK:")[1].strip()
                else:
                    abc_task = remaining.strip()
        
        if not abc_task or not emotion_task:
            # Fallback: use original prompt for both, but this shouldn't happen
            abc_task = user_prompt
            emotion_task = user_prompt
        
        return abc_task, emotion_task
        
    except Exception as e:
        print(f"Error splitting tasks: {e}")
        # Fallback: return original prompt for both
        return user_prompt, user_prompt

def agent_D_aggregator(answer_B=None, answer_C=None):
    text = ""

    if answer_B:
        text += f"🎼 **ABC Score Expert Answer:**\n{answer_B}\n\n"

    if answer_C:
        text += f"🎵 **Emotion Expert Answer:**\n{answer_C}\n\n"

    if not text:
        text = "No specialized agent was required. No additional information."

    return text


def run_agent_system(user_prompt):
    """
    Main entry point for the multi-agent system.
    First validates input, then routes to appropriate agents.
    """
    # Step 0: Validate input (check ABC score and question)
    print("Validating input...")
    is_valid, error_message, verified_abc = validate_input(user_prompt)
    
    if not is_valid:
        return f"❌ Input Validation Error:\n{error_message}\n\nPlease correct your input and try again."
    
    if verified_abc:
        print(f"✓ ABC score validated ({len(verified_abc)} characters)")
    
    # Step 1: Controller decides which agents to use
    decision = agent_A_controller(user_prompt)
    print("Controller decision:", decision)

    answer_B = None
    answer_C = None

    if decision == "BOTH":
        # Split the task into ABC-related and Emotion-related parts
        print("Splitting task for both agents...")
        abc_task, emotion_task = split_tasks_for_agents(user_prompt)
        print(f"ABC Task: {abc_task[:100]}...")
        print(f"Emotion Task: {emotion_task[:100]}...")
        
        # Give each agent their specific task
        answer_B = agent_B_abc_system(abc_task)
        answer_C = agent_C_emotion_system(emotion_task)
        
    elif decision == "ABC":
        answer_B = agent_B_abc_system(user_prompt)

    elif decision == "EMOTION":
        answer_C = agent_C_emotion_system(user_prompt)
    
    # If decision is "NONE", both answers remain None

    final_answer = agent_D_aggregator(answer_B, answer_C)
    return final_answer


# Main program entry point
if __name__ == "__main__":
    import sys
    
    # Check if running in batch mode (with CSV file) or interactive mode
    if len(sys.argv) > 1:
        # Batch mode: process CSV file
        csv_path = sys.argv[1]
        print(f"Processing CSV file: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Check if CSV has required columns
        if "prompt" not in df.columns:
            print("Error: CSV file must have a 'prompt' column")
            sys.exit(1)
        
        results = []
        for i, row in df.iterrows():
            print(f"\nProcessing sample {i+1}/{len(df)}...")
            user_prompt = row["prompt"]
            
            try:
                answer = run_agent_system(user_prompt)
                results.append(answer)
                print(f"Answer: {answer[:100]}...")  # Print first 100 chars
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                results.append(f"Error: {str(e)}")
        
        # Save results
        df["agent_answer"] = results
        output_path = csv_path.replace(".csv", "_multi_agent_results.csv")
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
    else:
        # Interactive mode
        print("=" * 60)
        print("Multi-Agent System for Symbolic Music Understanding")
        print("=" * 60)
        print("\nEnter your question about ABC notation or emotion classification.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                answer = run_agent_system(user_input)
                print(f"\nSystem: {answer}\n")
            except Exception as e:
                print(f"\nError: {e}\n")

