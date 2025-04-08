# def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
#     raise NotImplementedError()

# if __name__ == "__main__":
#     from fire import Fire

#     Fire(generate_dataset)


import time
import json
from pathlib import Path

from tqdm import tqdm
import torch

from .cot import CoTModel
from .data import Dataset
from .base_llm import BaseLLM


def parse_answer(answer: str) -> float:
    """
    Extract the float inside <answer>...</answer> or fallback to last number in the string.
    """
    import re
    try:
        if "<answer>" in answer and "</answer>" in answer:
            return float(answer.split("<answer>")[1].split("</answer>")[0].replace(",", "").strip())
    except Exception:
        pass
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", answer.replace(",", ""))
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return float("nan")
    return float("nan")


# def is_correct(pred: float, true: float, tol: float = 1e-2) -> bool:
#     if pred != pred:  # NaN check
#         return False
#     return abs(pred - true) < tol

def is_correct(pred: float, true: float, relative_tolerance: float = 0.05) -> bool:
    try:
        return abs(round(pred, 3) - round(true, 3)) < relative_tolerance * abs(round(true, 3))
    except:
        return False


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    # Confirm device
    print("🔍 PyTorch device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("✅ Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    model = CoTModel()
    dataset = Dataset("train")
    total = len(dataset.data)
    print(f"\n Dataset size: {total} examples\n")

    saved = []
    start_time = time.time()

    for idx, (question, true_answer) in enumerate(dataset.data):
        prompt = model.format_prompt(question)
        generations = model.batched_generate(
            [prompt], num_return_sequences=oversample, temperature=temperature
        )[0]

        found = False
        for i, reasoning in enumerate(generations):
            pred = parse_answer(reasoning)
            if is_correct(pred, true_answer):
                saved.append([question, true_answer, reasoning])
                # print(f"[✓] Example {idx+1}/{total} | Match on sample {i+1}/{oversample}")
                found = True
                break

        if not found:
            continue 
            # print(f"[✗] Example {idx+1}/{total} | No correct match found.")

        # Time estimate every 10 samples
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            percent_done = (idx + 1) / total
            est_total_time = elapsed / percent_done
            time_left = est_total_time - elapsed
            print(f"⏱️  Progress: {idx + 1}/{total} ({percent_done:.1%}) — ETA: {time_left/60:.1f} min")
            print(f'num saved: {len(saved)}: {round(len(saved) / (idx+1),2)}%')
            print(f'last example:')
            print(saved[-1])
            print()

    print(f"\n Done! Collected {len(saved)} valid samples from {total} examples.")

    # Save output
    output_path = Path(__file__).parent / "data" / output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(saved, f, indent=2)

    print(f"📄 Saved to {output_path.resolve()}")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)