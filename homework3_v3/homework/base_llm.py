from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        return question

    # def parse_answer(self, answer: str) -> float:
    #     """
    #     Parse the <answer></answer> tag and return a float.
    #     This function is somewhat robust to output errors (e.g. missing </answer> tags).
    #     """
    #     try:
    #         return float(answer.split("<answer>")[1].split("</answer>")[0])
    #     except (IndexError, ValueError):
    #         return float("nan")


    # Updated def parse (using chatgpt to support)
    def parse_answer(self, answer: str) -> float:
        """
        Parse the output from the LLM and return a float.
        This version first tries to extract the number from within <answer></answer> tags.
        If not found, it attempts to extract the final number in the output text.
        """
        try:
            # Try to extract using the <answer> tags
            if "<answer>" in answer and "</answer>" in answer:
                num_str = answer.split("<answer>")[1].split("</answer>")[0]
                return float(num_str.replace(",", "").strip())
        except (IndexError, ValueError):
            pass

        # If the above fails, try to extract the last number in the output using regex.
        # This regex finds the last occurrence of a number (integer or decimal) in the string.
        number_matches = re.findall(r'(-?\d+(?:\.\d+)?)', answer.replace(",", ""))
        if number_matches:
            try:
                return float(number_matches[-1].strip())
            except ValueError:
                return float("nan")
        
        return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.

        # -- Previous code -- 

        # micro_batch_size = 32
        # if len(prompts) > micro_batch_size:
        #     return [
        #         r
        #         for idx in tqdm(
        #             range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
        #         )
        #         for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
        #     ]

        # raise NotImplementedError()

        # -- Updated code -- 
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            results = []
            for idx in tqdm(range(0, len(prompts), micro_batch_size),
                            desc=f"LLM Running on Micro Batches {micro_batch_size}"):
                batch = prompts[idx: idx + micro_batch_size]
                results.extend(self.batched_generate(batch, num_return_sequences, temperature))
            return results

        # Ensure left-padding
        self.tokenizer.padding_side = "left"

        # Tokenize and move tensors to device
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        # Determine number of sequences to generate per prompt
        n_return = num_return_sequences if num_return_sequences is not None else 1

        # Setup generation parameters
        do_sample = temperature > 0

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=n_return,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Only decode the new tokens after the original prompt.
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # If only one sequence per prompt, return a flat list of strings.
        if num_return_sequences is None:
            return decoded
        else:
            # Otherwise, group the outputs into a list for each prompt.
            num_prompts = len(prompts)
            return [decoded[i * n_return:(i + 1) * n_return] for i in range(num_prompts)]



    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)

        for q, raw in zip(questions, generations):
            print("----- Raw Output -----")
            print("Question:", q)
            print("Raw Generation:", raw)
            print("parsed answer:", self.parse_answer(raw))
            print("----------------------")

        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
