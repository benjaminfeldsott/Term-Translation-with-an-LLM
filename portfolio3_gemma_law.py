# Portfolio Task 3: term translation with Gemma on the Shaoxing law corpus (Chinese -> English)
# Prompts 1-3 from Kostikova et al. (2023), run once each over the same selected segments
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# 1. LM Studio local server (same setup as bear-trap)
API_BASE = "http://localhost:1235/v1"
MODEL = "gemma-4-12b-it-qat"      # name shown under the model in My Models (CORSAIR models directory)
TEMPERATURE = 0.0          # same settings for every call, so only the prompt changes
REQUEST_TIMEOUT = 900      # seconds per segment; the first call also waits for LM Studio to load Gemma
MAX_FAILURES = 5           # stop if the server keeps failing
MAX_ANSWER_TOKENS = 512    # longest reference is about 100 words; stops runaway commentary

# Gemma's thinking markers, in case the reasoning text comes back inside the answer
THOUGHT_START = "<|channel>thought"
THOUGHT_END = "<channel|>"

# 2. ENTER YOUR PATHS HERE PLEASE
OUTPUT_FOLDER = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_FOLDER / "portfolio3_law_translations.json"

SOURCE_LANGUAGE = "Chinese"
TARGET_LANGUAGE = "English"
DOMAIN = "legal"

# 3. Output rules, placed at the top of all three prompts so every answer is a single clean translation
OUTPUT_RULES = (
    f"Respond with only the {TARGET_LANGUAGE} translation of the text below. "
    f"Give exactly one translation. "
    f"Do not give alternative translations, explanations, notes, comments, vocabulary lists, "
    f"romanization, headings, quotation marks or any formatting."
)

# The JSON holds 51 segments, one from each of 51 laws in 15_Law_Corpus_Mainland,
# each with its source, reference, Google translation and term list.
# Term list = each Chinese word aligned to the English the reference uses for it (see README),
# both sides copied verbatim from that segment (checked when the file loads).


# The three prompts, worded as in the assignment brief, with the output rules on top and the segment after
def prompt_1(record):
    return (
        f"{OUTPUT_RULES}\n\n"
        f"Translate the sentences below from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}\n\n"
        f"{clean_source(record['source'])}"
    )


def prompt_2(record):
    return (
        f"{OUTPUT_RULES}\n\n"
        f"You are a professional translator who is asked to translate documents from the {DOMAIN} domain. "
        f"Translate the sentences below from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}\n\n"
        f"{clean_source(record['source'])}"
    )


def prompt_3(record):
    term_lines = "\n".join(f"{term['source']} ## {term['target']}" for term in record["terms"])
    return (
        f"{OUTPUT_RULES}\n\n"
        f"You are a professional translator who is asked to translate documents from the {DOMAIN} domain. "
        f"You are given the following list of term from the {DOMAIN} domain with their translations. "
        f"The source and target terms are separated by the characters ##\n\n"
        f"-----Start of list of terms----\n\n"
        f"{term_lines}\n\n"
        f"-----End of list of terms----\n\n"
        f"Given this list of terms, translate the sentences below from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}. "
        f"Make sure the terms are translated according to the list provided.\n\n"
        f"{clean_source(record['source'])}"
    )


PASSES = [
    ("target1", prompt_1),
    ("target2", prompt_2),
    ("target3", prompt_3),
]


def clean_source(text):
    # a few scraped rows end in a stray replacement character
    return text.replace("�", "").strip()


def save_document(records, filepath):
    # write to a temp file first so a crash never leaves half a JSON file
    tmp_path = filepath.with_name(filepath.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as output:
        json.dump(records, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(tmp_path, filepath)


def load_document(filepath):
    records = json.loads(filepath.read_text(encoding="utf-8"))
    for record in records:
        for term in record["terms"]:
            if term["source"] not in record["source"]:
                raise ValueError(f"segment {record['sentence_id']}: '{term['source']}' not in source")
            if term["target"] not in record["reference"]:
                raise ValueError(f"segment {record['sentence_id']}: '{term['target']}' not in reference")
    return records


def check_answer(text, finish_reason):
    # the answer is stored exactly as Gemma gave it; these flags mark anything that is not a clean translation
    problems = []
    if finish_reason == "length":
        problems.append("cut off at the token limit")
    if "\n" in text:
        problems.append("more than one line")
    if re.search(r"\*\*|#|^>|^\s*[-*] |option", text, re.IGNORECASE | re.MULTILINE):
        problems.append("formatting or alternative translations")
    if re.search(r"[\u4e00-\u9fff]", text):
        problems.append("contains Chinese characters")
    return problems


def ask_gemma(prompt_text):
    # one closed dialogue: a single user message, no system message, no earlier turns
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_ANSWER_TOKENS,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        answer = json.loads(response.read().decode("utf-8"))
    choice = answer["choices"][0]
    translation = choice["message"].get("content") or ""

    # Gemma 4 thinks first; LM Studio normally moves that to reasoning_content, but drop any leftover
    if THOUGHT_END in translation:
        translation = translation.split(THOUGHT_END)[-1]
    translation = translation.replace(THOUGHT_START, "").strip()
    if not translation:
        raise ValueError("empty answer (the model may have spent its whole reply thinking)")
    return translation, choice.get("finish_reason")


def run_pass(records, target_field, prompt_builder):
    # one full pass over the document per prompt, one segment at a time, never in parallel
    missing = [record for record in records if not record[target_field]]
    print(f"{target_field}: {len(records) - len(missing)} done, {len(missing)} to go")
    failures = 0
    for record in missing:
        # each segment is enclosed: create the prompt, run one dialogue, record the answer, delete
        prompt_text = prompt_builder(record)
        try:
            response_text, finish_reason = ask_gemma(prompt_text)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
            del prompt_text
            failures += 1
            print(f"  segment {record['sentence_id']} failed: {error}")
            if failures >= MAX_FAILURES:
                print("  server keeps failing - stopping. Rerun to pick up where it left off.")
                return False
            time.sleep(3)
            continue
        record[target_field] = response_text
        record.setdefault("checks", {})[target_field] = check_answer(response_text, finish_reason)
        save_document(records, OUTPUT_FILE)
        del prompt_text, response_text, finish_reason
        failures = 0
    return True


def main():
    records = load_document(OUTPUT_FILE)
    print(f"Loaded {len(records)} segments from {OUTPUT_FILE.name}")

    for target_field, prompt_builder in PASSES:
        if not run_pass(records, target_field, prompt_builder):
            break

    still_empty = sum(1 for record in records for field, _ in PASSES if not record[field])
    flagged = sum(1 for record in records for problems in record.get("checks", {}).values() if problems)
    print(f"Finished. Empty targets left: {still_empty}. Flagged answers: {flagged}")


if __name__ == "__main__":
    main()
