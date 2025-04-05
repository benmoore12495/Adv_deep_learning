from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        # raise NotImplementedError()
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                  "You are a helpful AI assist for math conversions. Think through your answers."
                  "ALWAYS provide the final answer at the end enclosed in <answer> and </answer> tags."
                )
                
            },
            {
                "role": "user",
                "content": "Convert 2 hours to s."
            },
            {
                "role": "assistant",
                "content":"Each hour is 3600 seconds. The question asks for 2 hours, so 2 * 3600 = 7200: <answer>7200</answer>"
            },
            {
                "role": "user",
                "content": "Convert 5 hours to m"
            },
            {
                "role": "assistant",
                "content":"Each hour is 60 minutes. The question asks for 5 hours so 5 * 60 = 300: <answer>300</answer>"
            },
            {
                "role": "user",
                "content": "Express 3 ton as a quantity of kg"
            },
            {
                "role": "assistant",
                "content":"1 ton = 1000 but the question asks for 3 tons, so 3 * 1000 = 3000:  <answer>3000</answer>"
            },
            {
                "role": "user",
                "content": "What is the conversion from month to h for 3 units?"
            },
            {
                "role": "assistant",
                "content":"3 units = 3 months in this question. There are 24 hours in a day and 30 days in a month. So 24*30*3 = 2160:  <answer>2160</answer>"
            },
            {
                "role": "user",
                "content": question
            }

        ]
        
        # Convert the chat dialogue to the final prompt using the chat template.
        final_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # print()
        # print()
        # print('----------------------')
        # print('----------------------')
        # print('----------------------')
        # print(final_prompt)
        return final_prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
