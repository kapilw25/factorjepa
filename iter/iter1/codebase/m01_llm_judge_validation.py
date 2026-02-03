"""
Step 0: Validate LLM-as-Judge Concept
=====================================
Objective: Prove that an LLM can act as a path quality judge

This script tests whether LLMs can evaluate navigation paths by:
1. Comparing two paths (efficient vs inefficient)
2. Scoring them on efficiency and directness
3. Explaining the reasoning

Usage:
    python m01_llm_judge_validation.py                    # Run all tests
    python m01_llm_judge_validation.py --dry-run          # Show prompts only
    python m01_llm_judge_validation.py --model gpt-4o     # Use specific model
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

# API Key (loaded from environment)
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "step0"
TEST_CASES_FILE = DATA_DIR / "test_cases.json"
PROMPTS_FILE = DATA_DIR / "prompts.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step0"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_cases() -> dict:
    """Load test cases from JSON file"""
    with open(TEST_CASES_FILE, "r") as f:
        return json.load(f)


def load_prompts() -> dict:
    """Load prompt templates from JSON file"""
    with open(PROMPTS_FILE, "r") as f:
        return json.load(f)


# ============================================================================
# OPENAI API CLIENT
# ============================================================================

def call_openai_api(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Call OpenAI API for LLM-as-Judge

    Args:
        prompt: The prompt to send
        model: Model to use (gpt-4o-mini, gpt-4o, gpt-4-turbo)

    Returns:
        Response text from the model
    """
    import requests

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPEN_API_KEY not found in environment variables.\n"
            "Please check your .env file at: " + str(env_path)
        )

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert judge for evaluating navigation paths. Always respond in valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1  # Low temperature for consistent evaluation
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def build_judge_prompt(test_case: dict, prompt_template: str) -> str:
    """Build the prompt for the judge model using template"""

    # Format the prompt with test case data
    return prompt_template.format(
        task=test_case["task"],
        path_a_actions=test_case["path_a"]["actions"],
        path_a_length=test_case["path_a"]["length"],
        path_b_actions=test_case["path_b"]["actions"],
        path_b_length=test_case["path_b"]["length"]
    )


def build_judge_prompt_with_gt(test_case: dict, prompt_template: str) -> str:
    """Build the prompt with ground truth included"""

    # Get room layout as string
    room_layout = "\n".join(test_case["room_layout"]["ascii_map"])

    return prompt_template.format(
        task=test_case["task"],
        room_layout=room_layout,
        optimal_path=test_case["ground_truth"]["optimal_path"],
        optimal_length=test_case["ground_truth"]["optimal_length"],
        path_a_actions=test_case["path_a"]["actions"],
        path_a_length=test_case["path_a"]["length"],
        path_b_actions=test_case["path_b"]["actions"],
        path_b_length=test_case["path_b"]["length"]
    )


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_judge_response(response: str) -> dict:
    """Parse the JSON response from the judge"""
    try:
        # Try to extract JSON from the response
        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            # Find JSON object directly
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
            else:
                json_str = response

        return json.loads(json_str)

    except json.JSONDecodeError as e:
        return {
            "raw_response": response,
            "parse_error": True,
            "error_message": str(e)
        }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_test_case(
    test_case: dict,
    prompt_template: str,
    model: str = "gpt-4o-mini",
    use_ground_truth: bool = False
) -> dict:
    """Run evaluation for a single test case"""

    # Build prompt
    if use_ground_truth:
        prompt = build_judge_prompt_with_gt(test_case, prompt_template)
    else:
        prompt = build_judge_prompt(test_case, prompt_template)

    print(f"\n{'='*60}")
    print(f"Test: {test_case['id']}")
    print(f"Task: {test_case['task']}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    # Show paths
    print(f"\nPath A ({test_case['path_a']['length']} steps): {test_case['path_a']['actions']}")
    print(f"Path B ({test_case['path_b']['length']} steps): {test_case['path_b']['actions']}")
    print(f"Ground Truth ({test_case['ground_truth']['optimal_length']} steps): {test_case['ground_truth']['optimal_path']}")

    # Call API
    print("\nCalling OpenAI API...")
    response = call_openai_api(prompt, model=model)

    # Parse response
    result = parse_judge_response(response)

    # Check if winner matches expected
    if "winner" in result and not result.get("parse_error"):
        expected = test_case["expected_winner"]
        actual = result["winner"]
        is_correct = expected.lower().replace(" ", "") in actual.lower().replace(" ", "")

        print(f"\n--- RESULTS ---")
        print(f"Expected Winner: {expected}")
        print(f"LLM's Choice: {actual}")
        print(f"Correct: {'✅ YES' if is_correct else '❌ NO'}")

        if "path_a" in result:
            print(f"\nPath A Scores: {result['path_a']}")
        if "path_b" in result:
            print(f"Path B Scores: {result['path_b']}")
        if "winner_reasoning" in result:
            print(f"\nReasoning: {result['winner_reasoning']}")

        result["is_correct"] = is_correct
        result["expected_winner"] = expected
    else:
        print(f"\n⚠️ Parse Error - Raw Response:\n{response[:500]}...")
        result["is_correct"] = False

    return result


def run_validation(
    model: str = "gpt-4o-mini",
    use_ground_truth: bool = False,
    prompt_version: str = "llm_judge_v1"
):
    """Run all test cases and summarize results"""

    # Load data
    test_data = load_test_cases()
    prompts_data = load_prompts()

    # Get prompt template
    if use_ground_truth:
        prompt_version = "llm_judge_v2_with_gt"

    prompt_template = prompts_data["prompts"][prompt_version]["template"]

    print("\n" + "="*60)
    print("STEP 0: LLM-AS-JUDGE VALIDATION")
    print("="*60)
    print(f"Model: {model}")
    print(f"Prompt Version: {prompt_version}")
    print(f"Include Ground Truth: {use_ground_truth}")
    print(f"Test Cases: {len(test_data['test_cases'])}")

    results = []
    correct_count = 0

    for test_case in test_data["test_cases"]:
        try:
            result = evaluate_test_case(
                test_case,
                prompt_template,
                model=model,
                use_ground_truth=use_ground_truth
            )
            results.append({
                "test_id": test_case["id"],
                "task": test_case["task"],
                "result": result
            })
            if result.get("is_correct", False):
                correct_count += 1
        except Exception as e:
            print(f"\n❌ Error in {test_case['id']}: {e}")
            results.append({
                "test_id": test_case["id"],
                "error": str(e)
            })

    # Summary
    total = len(test_data["test_cases"])
    accuracy = correct_count / total * 100 if total > 0 else 0

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total Tests: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.1f}%")

    if accuracy >= 100:
        print("\n✅ PERFECT! LLM correctly judged all paths!")
        print("   → Ready to proceed to Phase 1")
    elif accuracy >= 70:
        print("\n✅ SUCCESS: LLM can act as a path quality judge!")
        print("   → Ready to proceed to Phase 1 (may need prompt refinement)")
    else:
        print("\n⚠️  NEEDS WORK: Consider refining prompts or using stronger model")

    return {
        "model": model,
        "prompt_version": prompt_version,
        "total_tests": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate LLM-as-Judge concept for path evaluation"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--with-gt",
        action="store_true",
        help="Include ground truth in prompt"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling API"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - Showing prompts only\n")
        test_data = load_test_cases()
        prompts_data = load_prompts()

        prompt_version = "llm_judge_v2_with_gt" if args.with_gt else "llm_judge_v1"
        prompt_template = prompts_data["prompts"][prompt_version]["template"]

        for test_case in test_data["test_cases"]:
            print(f"{'='*60}")
            print(f"Test: {test_case['id']}")
            print(f"{'='*60}")

            if args.with_gt:
                prompt = build_judge_prompt_with_gt(test_case, prompt_template)
            else:
                prompt = build_judge_prompt(test_case, prompt_template)

            print(prompt)
            print("\n")
    else:
        # Run validation
        results = run_validation(
            model=args.model,
            use_ground_truth=args.with_gt
        )

        # Save results if requested
        if args.save:
            OUTPUT_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"validation_{args.model}_{timestamp}.json"

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
