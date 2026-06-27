import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


INPUT_DS_PATH = "data/raw/the_stack_libs_code_only"
OUTPUT_DS_PATH = "data/synthetic_docstring_ds"

BATCH_SIZE = 10
MAX_CONCURRENT = 15
MAX_RETRIES = 3

MODEL = "mistralai/mistral-small-2603"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
EOS_TOKEN: str = os.environ.get("TOKENIZER_EOS_TOKEN", "")


SYSTEM_PROMPT = """
    You are a professional Python docstring generator
    You will receive a JSON object whose keys are string indices and whose value
    are Python code blocks (functions, methods, or classes)
    Return ONLY a valid JSON object with the SAME keys and the correspondin
    docstring summarizing the code block as the value for each key
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
) -> dict[str, str] | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(batch_payload, ensure_ascii=False)},
    ]
    request_body = {
        "model": MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": True},
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
                return docstrings

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
        return None


async def _process_batch(
    batch_indices: list[int],
    batch_texts: list[str],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    results: dict[int, str],
    failed_indices: list[int],
    eos_token: str,
) -> None:
    str_keys = [str(idx) for idx in batch_indices]
    batch_payload = dict(zip(str_keys, batch_texts))

    docstrings = await _call_api(client, batch_payload, semaphore)

    if docstrings is None:
        log.error(
            "Skipping %d samples (indices: %s).", len(batch_indices), batch_indices
        )
        failed_indices.extend(batch_indices)
        return

    for global_idx, code_text in zip(batch_indices, batch_texts):
        str_idx = str(global_idx)
        docstring = docstrings.get(str_idx, "")
        if not docstring:
            log.warning(
                "No docstring returned for index %d — marking as failed.", global_idx
            )
            failed_indices.append(global_idx)
            continue
        results[global_idx] = f"{code_text}{docstring}{eos_token}"


async def main(eos_token: str = EOS_TOKEN) -> None:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Export it as an environment variable before running this script.\n"
            "  export OPENROUTER_API_KEY=sk-or-..."
        )

    log.info("Loading dataset from '%s' …", INPUT_DS_PATH)
    ds: Dataset = Dataset.load_from_disk(INPUT_DS_PATH)
    texts: list[str] = ds["text"]
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

    results: dict[int, str] = {}
    failed_indices: list[int] = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [
            _process_batch(
                idx_list,
                text_list,
                client,
                semaphore,
                results,
                failed_indices,
                eos_token,
            )
            for idx_list, text_list in batches
        ]
        await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t0
    log.info(
        "Pipeline finished in %.1fs — %d succeeded, %d failed.",
        elapsed,
        len(results),
        len(failed_indices),
    )

    merged_texts: list[str] = [results[idx] for idx in all_indices if idx in results]
    log.info("Output dataset will contain %d samples.", len(merged_texts))

    out_path = Path(OUTPUT_DS_PATH)
    out_path.mkdir(parents=True, exist_ok=True)

    out_ds = Dataset.from_dict({"text": merged_texts})
    out_ds.save_to_disk(str(out_path))
    log.info("Dataset saved to '%s'.", out_path)

    if failed_indices:
        failed_log = out_path / "failed_indices.json"
        with open(failed_log, "w") as f:
            json.dump(failed_indices, f, indent=2)
        log.warning(
            "%d sample(s) failed and were omitted. Indices logged to '%s'.",
            len(failed_indices),
            failed_log,
        )
    else:
        log.info("All batches completed successfully — no failures.")


if __name__ == "__main__":
    asyncio.run(main(eos_token=EOS_TOKEN))
