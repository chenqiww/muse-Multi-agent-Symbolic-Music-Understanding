"""
Test script for controller accuracy.
Tests the controller function from multi_agent_system.py
using the generated_dataset.csv data.
Assumes all samples have ABC score and questions (validation passed).
"""

import pandas as pd
from multi_agent_system import agent_A_controller
import sys

def test_controller_accuracy(df):
    """
    Test controller accuracy.
    Assumes all samples have ABC score and questions (validation passed).
    
    Controller returns: ABC, EMOTION, BOTH, or NONE
    We test:
    1. Whether controller correctly identifies emotion questions (ans[1])
    2. Whether controller correctly identifies ABC/QA-related questions 
       Note: signature key (ans[2]) and bars (ans[3]) are both ABC/QA questions, 
       so we combine them - if either is asked, it's an ABC-related question.
    """
    print("=" * 60)
    print("Testing Controller Accuracy")
    print("=" * 60)
    
    correct_emotion = 0
    correct_abc_related = 0  # key or bars questions are ABC-related
    total = 0
    skipped = 0
    
    results = []
    error_records = []  # Store error records
    
    for idx, row in df.iterrows():
        # Handle NaN values
        question = row['question'] if pd.notna(row['question']) else ""
        ans = str(row['ans']).zfill(4) if pd.notna(row['ans']) else "0000"
        
        # Check if sample has ABC score and at least one question
        has_abc = int(ans[0])
        asked_emotion_gt = int(ans[1])
        asked_key_gt = int(ans[2])
        asked_bars_gt = int(ans[3])
        has_any_question = 1 if (asked_emotion_gt == 1 or asked_key_gt == 1 or asked_bars_gt == 1) else 0
        
        # Skip if no ABC score or no questions
        if has_abc == 0 or has_any_question == 0:
            skipped += 1
            results.append({
                'idx': idx,
                'decision': 'SKIPPED',
                'emotion_gt': asked_emotion_gt,
                'emotion_pred': -1,
                'emotion_correct': None,
                'key_gt': asked_key_gt,
                'bars_gt': asked_bars_gt,
                'abc_related_gt': 1 if (asked_key_gt == 1 or asked_bars_gt == 1) else 0,
                'abc_related_pred': -1,
                'abc_related_correct': None,
                'skip_reason': 'No ABC score' if has_abc == 0 else 'No questions'
            })
            continue
        
        # Ground truth: which questions were asked
        # ans[1] = emotion, ans[2] = signature key, ans[3] = bars
        # Note: signature key and bars are both ABC/QA questions, so we combine them
        # ABC/QA-related: either signature key OR bars question (both are ABC/QA type)
        asked_abc_related_gt = 1 if (asked_key_gt == 1 or asked_bars_gt == 1) else 0
        
        # Run controller
        try:
            decision = agent_A_controller(question)
            
            # Parse controller decision
            # Check if controller detected emotion question
            asked_emotion_pred = 1 if "EMOTION" in decision or "BOTH" in decision else 0
            
            # Check if controller detected ABC-related question (key/bars are ABC-related)
            asked_abc_related_pred = 1 if "ABC" in decision or "BOTH" in decision else 0
            
            # Compare with ground truth
            emotion_correct = (asked_emotion_pred == asked_emotion_gt)
            abc_related_correct = (asked_abc_related_pred == asked_abc_related_gt)
            
            correct_emotion += emotion_correct
            correct_abc_related += abc_related_correct
            total += 1
            
            results.append({
                'idx': idx,
                'decision': decision,
                'emotion_gt': asked_emotion_gt,
                'emotion_pred': asked_emotion_pred,
                'emotion_correct': emotion_correct,
                'key_gt': asked_key_gt,
                'bars_gt': asked_bars_gt,
                'abc_related_gt': asked_abc_related_gt,
                'abc_related_pred': asked_abc_related_pred,
                'abc_related_correct': abc_related_correct
            })
            
            if not (emotion_correct and abc_related_correct):
                # Record error
                error_records.append({
                    'idx': idx,
                    'decision': decision,
                    'emotion_gt': asked_emotion_gt,
                    'emotion_pred': asked_emotion_pred,
                    'emotion_correct': emotion_correct,
                    'abc_related_gt': asked_abc_related_gt,
                    'abc_related_pred': asked_abc_related_pred,
                    'abc_related_correct': abc_related_correct,
                    'key_gt': asked_key_gt,
                    'bars_gt': asked_bars_gt,
                    'question': question if question else ""  # Store full question
                })
                print(f"❌ Row {idx}: Decision={decision}")
                print(f"   Emotion: GT={asked_emotion_gt}, Pred={asked_emotion_pred} {'✓' if emotion_correct else '✗'}")
                print(f"   ABC/QA-related (signature key or bars): GT={asked_abc_related_gt}, Pred={asked_abc_related_pred} {'✓' if abc_related_correct else '✗'}")
                print(f"   (Signature Key={asked_key_gt}, Bars={asked_bars_gt} - both are ABC/QA type)")
                if question:
                    print(f"   Question: {str(question)[:80]}...")
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            total += 1
            error_msg = str(e)
            results.append({
                'idx': idx,
                'decision': 'ERROR',
                'emotion_gt': asked_emotion_gt,
                'emotion_pred': -1,
                'emotion_correct': False,
                'key_gt': asked_key_gt,
                'bars_gt': asked_bars_gt,
                'abc_related_gt': asked_abc_related_gt,
                'abc_related_pred': -1,
                'abc_related_correct': False
            })
            # Record error
            error_records.append({
                'idx': idx,
                'decision': 'ERROR',
                'emotion_gt': asked_emotion_gt,
                'emotion_pred': -1,
                'emotion_correct': False,
                'abc_related_gt': asked_abc_related_gt,
                'abc_related_pred': -1,
                'abc_related_correct': False,
                'key_gt': asked_key_gt,
                'bars_gt': asked_bars_gt,
                'question': question if question else "",
                'error': error_msg
            })
    
    emotion_acc = (correct_emotion / total) * 100 if total > 0 else 0
    abc_related_acc = (correct_abc_related / total) * 100 if total > 0 else 0
    
    print(f"\nController Accuracy (tested {total} cases, skipped {skipped} cases without ABC score or questions):")
    print(f"  Emotion detection: {correct_emotion}/{total} = {emotion_acc:.2f}%")
    print(f"  ABC/QA-related detection (signature key or bars): {correct_abc_related}/{total} = {abc_related_acc:.2f}%")
    if total > 0:
        print(f"  Overall (both correct): {(sum(1 for r in results if r.get('emotion_correct') and r.get('abc_related_correct')) / total * 100):.2f}%")
    
    return results, emotion_acc, abc_related_acc, error_records


def main():
    # Load dataset
    dataset_path = 'data/generated_dataset.csv'
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path, dtype={'ans': str})
        print(f"Loaded {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Ensure ans column has 4 digits
    df['ans'] = df['ans'].astype(str).str.zfill(4)
    
    # Test controller (assumes all samples have ABC score and questions)
    ctrl_results, emotion_acc, abc_related_acc, error_records = test_controller_accuracy(df)
    
    # Save results
    ctrl_df = pd.DataFrame(ctrl_results)
    
    output_path = 'data/controller_test_results.csv'
    
    # Merge results
    combined_data = []
    for idx in df.index:
        ctrl_row = ctrl_df[ctrl_df['idx'] == idx].iloc[0] if len(ctrl_df[ctrl_df['idx'] == idx]) > 0 else None
        
        row_data = {
            'idx': idx,
            'question': df.loc[idx, 'question'] if pd.notna(df.loc[idx, 'question']) else "",
            'ans': str(df.loc[idx, 'ans']).zfill(4) if pd.notna(df.loc[idx, 'ans']) else "0000",
        }
        
        if ctrl_row is not None:
            row_data.update({
                'controller_decision': ctrl_row.get('decision', 'N/A'),
                'emotion_gt': ctrl_row.get('emotion_gt', -1),
                'emotion_pred': ctrl_row.get('emotion_pred', -1),
                'emotion_correct': ctrl_row.get('emotion_correct', None),
                'key_gt': ctrl_row.get('key_gt', -1),
                'bars_gt': ctrl_row.get('bars_gt', -1),
                'abc_related_gt': ctrl_row.get('abc_related_gt', -1),
                'abc_related_pred': ctrl_row.get('abc_related_pred', -1),
                'abc_related_correct': ctrl_row.get('abc_related_correct', None),
            })
        else:
            row_data.update({
                'controller_decision': 'N/A',
                'emotion_gt': -1,
                'emotion_pred': -1,
                'emotion_correct': None,
                'key_gt': -1,
                'bars_gt': -1,
                'abc_related_gt': -1,
                'abc_related_pred': -1,
                'abc_related_correct': None,
            })
        
        combined_data.append(row_data)
    
    combined_df = pd.DataFrame(combined_data)
    
    combined_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Controller Accuracy:")
    print(f"  Emotion: {emotion_acc:.2f}%")
    print(f"  ABC/QA-related (signature key or bars): {abc_related_acc:.2f}%")
    
    # Save error records to markdown
    if error_records:
        markdown_path = 'data/controller_test_errors.md'
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# Controller Test Error Records\n\n")
            f.write(f"Total Errors: {len(error_records)}\n\n")
            f.write("---\n\n")
            
            for i, err in enumerate(error_records, 1):
                f.write(f"## Error #{i} - Row {err['idx']}\n\n")
                f.write(f"**Controller Decision:** `{err['decision']}`\n\n")
                f.write("### Classification Results\n\n")
                f.write("| Type | Ground Truth | Predicted | Correct |\n")
                f.write("|------|--------------|-----------|----------|\n")
                f.write(f"| Emotion | {err['emotion_gt']} | {err['emotion_pred']} | {'✓' if err['emotion_correct'] else '✗'} |\n")
                f.write(f"| ABC/QA-related | {err['abc_related_gt']} | {err['abc_related_pred']} | {'✓' if err['abc_related_correct'] else '✗'} |\n\n")
                f.write(f"**Details:**\n")
                f.write(f"- Key Question: {err.get('key_gt', -1)}\n")
                f.write(f"- Bars Question: {err.get('bars_gt', -1)}\n\n")
                
                if err.get('question'):
                    f.write("### Question\n\n")
                    f.write("```\n")
                    f.write(f"{err['question']}\n")
                    f.write("```\n\n")
                
                if err.get('error'):
                    f.write("### Error Message\n\n")
                    f.write(f"```\n{err['error']}\n```\n\n")
                
                f.write("---\n\n")
        
        print(f"\nError records saved to: {markdown_path}")
        
        # Print error records summary
        print(f"\n" + "=" * 60)
        print(f"Error Records ({len(error_records)} errors)")
        print("=" * 60)
        for i, err in enumerate(error_records, 1):
            print(f"\nError #{i} - Row {err['idx']}:")
            print(f"  Controller Decision: {err['decision']}")
            print(f"  Emotion: GT={err['emotion_gt']}, Pred={err['emotion_pred']}, Correct={err['emotion_correct']}")
            print(f"  ABC/QA-related: GT={err['abc_related_gt']}, Pred={err['abc_related_pred']}, Correct={err['abc_related_correct']}")
            print(f"  (Key={err.get('key_gt', -1)}, Bars={err.get('bars_gt', -1)})")
            if err.get('question'):
                print(f"  Question: {err['question'][:100]}...")
            if err.get('error'):
                print(f"  Error: {err['error']}")
    else:
        print("\n✓ No errors found! All predictions are correct.")


if __name__ == "__main__":
    main()

