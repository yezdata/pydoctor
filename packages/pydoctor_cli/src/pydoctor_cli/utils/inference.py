from llama_cpp import Llama


SYSTEM_PROMPT = '''You are a professional Python docstring generator.
Analyze the code in 'TARGET CODE'. Use 'CONTEXT' only for reference regarding class attributes and structure.
Generate a concise, single-paragraph docstring summarizing ONLY the 'TARGET CODE'.
The docstring must consist only of descriptive sentences.

CRITICAL RULES:
1. Output ONLY the raw docstring text. Do not include triple quotes (""").
2. Do not include structured sections like Args, Returns, or Raises.
3. Do not include conversational filler, explanations, or markdown code blocks.
4. Focus strictly on the functionality of the 'TARGET CODE'.'''


def get_chat_template(sample: dict) -> str:
    final_output = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nCONTEXT\n{sample['context']}\n\nTARGET CODE\n{sample['target']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return final_output


def generate_docstring(llm: Llama, sample: dict) -> str:
    prompt = get_chat_template(sample)

    response = llm(
        prompt,
        temperature=0.7,
        top_k=50,
        top_p=0.90,
        max_tokens=256,
        stop=["<|im_end|>"],
    )

    return response["choices"][0]["text"]
