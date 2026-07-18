from llama_cpp import Llama


SYSTEM_PROMPT = '''You are a professional Python documentation expert.
Analyze the 'TARGET CODE' and using also the 'CONTEXT' provide a concise docstring containing a high-level summary of its purpose and logic.
Scope Definitions:
1. 'CONTEXT' is the surrounding code that provides additional information about the 'TARGET CODE'.
2. If 'TARGET CODE' is a class, the 'CONTEXT' are signatures of the class methods, if 'TARGET CODE' is a method, the 'CONTEXT' is the constructor of its class.

CRITICAL RULES:
1. Output ONLY the raw docstring text. Do NOT output any other conversation filler. Do not include triple quotes (""").
2. Describe the semantic purpose and architectural role of the TARGET CODE, not its internal state or attributes.
3. Do not include structured sections like Args, Returns, or Raises.
4. Focus on what the TARGET CODE achieves, do NOT describe its implementation details.
5. Do not list initialization variables or internal data structures.
'''


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
