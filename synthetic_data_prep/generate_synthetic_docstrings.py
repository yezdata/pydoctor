import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from datasets import Dataset
from dotenv import load_dotenv

from src.utils.extract_code_stack_libs import extract_code


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

EXTRACTED_CODE_PATH = "data/extracted/the_stack_libs_code"
BATCHES_DIR = "data/synthetic_batches"

BATCH_SIZE = 10
MAX_CONCURRENT = 5
MAX_RETRIES = 5

MODEL = "mistralai/mistral-small-2603"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EOS_TOKEN = os.environ.get("TOKENIZER_EOS_TOKEN", "")

TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0


SYSTEM_PROMPT = """
    You are a professional Python docstring generator
    You will receive a JSON object whose keys are string indices and whose value
    are Python code blocks (functions, methods, or classes)
    Return ONLY a valid JSON object with the SAME keys and the corresponding
    docstring summarizing the code block as the value for each key
    The style of the doctring have to be only sentences summarizing the code block
    Do NOT write: 
        Args:
        Returns:
        Raises:
        or any other sections, keep the docstring as a single paragraph
    Do NOT wrap the output in markdown fences. Do NOT add any extra text
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _make_batches(
    all_indices: list[int],
    texts: list[str],
    size: int,
) -> list[tuple[list[int], list[str]]]:
    return [
        (all_indices[i : i + size], texts[i : i + size])
        for i in range(0, len(all_indices), size)
    ]


async def _call_api(
    client: httpx.AsyncClient,
    batch_payload: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, str] | None, int, int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(batch_payload, ensure_ascii=False)},
    ]
    request_body = {
        "model": MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=request_body,
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()
                raw_content: str = data["choices"][0]["message"]["content"]
                docstrings = _extract_json(raw_content)
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                return docstrings, prompt_tokens, completion_tokens

            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                wait = 2**attempt
                log.warning(
                    "HTTP/network error on attempt %d/%d: %s — retrying in %ds",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                wait = 2**attempt
                log.warning(
                    "Response parse error on attempt %d/%d: %s — retrying in %ds",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

        log.error(
            "All %d retries exhausted for batch keys: %s",
            MAX_RETRIES,
            list(batch_payload.keys()),
        )
        return None, 0, 0


async def _process_batch(
    batch_idx: int,
    batch_indices: list[int],
    batch_texts: list[str],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    results: dict[int, str],
    failed_indices: list[int],
    eos_token: str,
    batches_dir: Path,
) -> None:
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS
    str_keys = [str(idx) for idx in batch_indices]
    batch_payload = dict(zip(str_keys, batch_texts))

    docstrings, prompt_tokens, completion_tokens = await _call_api(
        client, batch_payload, semaphore
    )
    TOTAL_INPUT_TOKENS += prompt_tokens
    TOTAL_OUTPUT_TOKENS += completion_tokens

    batch_results = {}
    batch_failed = []
    if docstrings is None:
        log.error(
            "Skipping %d samples (indices: %s).", len(batch_indices), batch_indices
        )
        batch_failed.extend(batch_indices)
    else:
        for global_idx, code_text in zip(batch_indices, batch_texts):
            str_idx = str(global_idx)
            docstring = docstrings.get(str_idx, "")
            if not docstring:
                log.warning(
                    "No docstring returned for index %d — marking as failed.",
                    global_idx,
                )
                batch_failed.append(global_idx)
                continue
            batch_results[global_idx] = f"{code_text}{docstring}{eos_token}"
            if global_idx % 10 == 0:
                log.info(
                    "Token usage so far — Input: %d, Output: %d",
                    TOTAL_INPUT_TOKENS,
                    TOTAL_OUTPUT_TOKENS,
                )

    batch_file = batches_dir / f"batch_{batch_idx}.json"
    batch_data = {
        "batch_idx": batch_idx,
        "results": {str(k): v for k, v in batch_results.items()},
        "failed_indices": batch_failed,
    }
    temp_file = batch_file.with_suffix(".tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        temp_file.replace(batch_file)
        log.info("Saved batch %d to '%s'.", batch_idx, batch_file)
    except Exception as e:
        log.error("Failed to save batch %d: %s", batch_idx, e)

    results.update(batch_results)
    failed_indices.extend(batch_failed)


async def main(eos_token: str = EOS_TOKEN) -> None:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Export it as an environment variable before running this script.\n"
            "  export OPENROUTER_API_KEY=sk-or-..."
        )

    extract_code(EXTRACTED_CODE_PATH)

    ds = Dataset.load_from_disk(EXTRACTED_CODE_PATH)
    texts = ds["code"]
    total = len(texts)
    log.info("Loaded %d samples.", total)

    all_indices = list(range(total))
    batches = _make_batches(all_indices, texts, BATCH_SIZE)
    log.info(
        "Created %d batches of up to %d samples (concurrency cap: %d).",
        len(batches),
        BATCH_SIZE,
        MAX_CONCURRENT,
    )

    batches_dir = Path(BATCHES_DIR)
    batches_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, str] = {}
    failed_indices: list[int] = []

    completed_batches = set()
    for batch_file in batches_dir.glob("batch_*.json"):
        try:
            with open(batch_file, "r", encoding="utf-8") as f:
                batch_data = json.load(f)
                b_idx = batch_data["batch_idx"]
                b_results = batch_data.get("results", {})
                b_failed = batch_data.get("failed_indices", [])

                for k, v in b_results.items():
                    results[int(k)] = v
                failed_indices.extend(b_failed)
                completed_batches.add(b_idx)
        except Exception as e:
            log.warning("Failed to load completed batch from '%s': %s", batch_file, e)

    if completed_batches:
        log.info(
            "Loaded %d completed batches from '%s'.",
            len(completed_batches),
            batches_dir,
        )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = []
        for batch_idx, (idx_list, text_list) in enumerate(batches):
            if batch_idx in completed_batches:
                continue
            tasks.append(
                _process_batch(
                    batch_idx,
                    idx_list,
                    text_list,
                    client,
                    semaphore,
                    results,
                    failed_indices,
                    eos_token,
                    batches_dir,
                )
            )
        if tasks:
            log.info("Processing %d remaining batches...", len(tasks))
            await asyncio.gather(*tasks)
        else:
            log.info("All batches were already processed.")

    elapsed = time.perf_counter() - t0
    log.info(
        "Pipeline finished in %.1fs — %d succeeded, %d failed.",
        elapsed,
        len(results),
        len(failed_indices),
    )
    log.info(
        "Total tokens used — Input: %d, Output: %d",
        TOTAL_INPUT_TOKENS,
        TOTAL_OUTPUT_TOKENS,
    )

    if failed_indices:
        failed_log = batches_dir / "failed_indices.json"
        unique_failed = sorted(list(set(failed_indices)))
        with open(failed_log, "w") as f:
            json.dump(unique_failed, f, indent=2)
        log.warning(
            "%d sample(s) failed and were omitted. Indices logged to '%s'.",
            len(unique_failed),
            failed_log,
        )
    else:
        log.info("All batches completed successfully — no failures.")


if __name__ == "__main__":
    asyncio.run(main(eos_token=EOS_TOKEN))
