#!/usr/bin/env python3
"""
Batch PDF → clinical feature extraction using Ollama.
"""

import os
import re
import time
import math
import json
import random
import hashlib
import logging
import warnings
import traceback
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import camelot
except Exception:
    camelot = None

try:
    import tabula
except Exception:
    tabula = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

REPO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PARENT_FOLDER = REPO_ROOT / "pdfs"

RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "cache_ollama"
OUTPUT_SUFFIX = "extracted_clinical_data_cloud.csv"

MODEL_NAME = "gpt-oss:120b-cloud"

CHUNK_SIZE = 5000
MAX_WORKERS = 2
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "typical_age_of_onset",
    "gender_predominance",
    "life_expectancy",
    "presenting_symptoms_signs",
    "key_investigations_for_confirmation",
    "clinical_presentation",
    "manageability",
    "treatability",
    "onset_age",
]

QUOTA_TRIGGER_FILE = None
TIMER_START = None
INTERACTIVE_INTERVAL_SECONDS = 30 * 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pdf_extractor")

try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except Exception:
    pass

warnings.filterwarnings("ignore", message="Cannot set non-stroke color")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
if pdfplumber:
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)
if camelot:
    logging.getLogger("camelot").setLevel(logging.ERROR)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return b.decode("latin-1", errors="ignore")
        except Exception:
            return str(b)


def atomic_write_text(path: Path, text: str, encoding="utf-8"):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding=encoding) as f:
        f.write(text)
    try:
        os.replace(temp_path, path)
    except PermissionError:
        time.sleep(0.2)
        try:
            if path.exists():
                os.remove(path)
            os.replace(temp_path, path)
        except Exception as e:
            raise PermissionError(f"Failed writing {path}: {e}")


def is_quota_error(stderr: str, stdout: str, rc: int) -> bool:
    text = (stderr or "") + " " + (stdout or "")
    t = text.lower()
    if "quota" in t:
        return True
    if "rate limit" in t:
        return True
    if "limit exceeded" in t:
        return True
    if "insufficient balance" in t:
        return True
    if "payment required" in t:
        return True
    return False


def interactive_window(reason: str, pdf_id: str, message: str = "") -> str:
    print("\n=====================================")
    if reason == "quota":
        print("=== OLLAMA QUOTA / RATE LIMIT DETECTED ===")
    else:
        print("=== 30-MINUTE CHECKPOINT ===")
    print(f"Current file: {pdf_id}")
    if message:
        print("Message:", message[:300])
    print("-------------------------------------")
    if reason == "quota":
        prompt_text = (
            "Switch Ollama account / plan / key now (if needed).\n"
            "Then press Enter to retry this chunk,\n"
            "or type 'skip' to skip this chunk,\n"
            "or type 'abort' to stop the whole script: "
        )
    else:
        prompt_text = (
            "30 minutes have passed.\n"
            "If you want, switch Ollama account / plan / key now.\n"
            "Press Enter to continue normally,\n"
            "or type 'skip' to skip this chunk,\n"
            "or type 'abort' to stop the whole script: "
        )
    ans = input(prompt_text).strip().lower()
    if ans == "skip":
        return "skip"
    if ans == "abort":
        return "abort"
    return "retry"


def maybe_time_prompt(pdf_id: str) -> str:
    global TIMER_START
    if TIMER_START is None:
        TIMER_START = time.time()
        return "continue"
    now = time.time()
    if now - TIMER_START >= INTERACTIVE_INTERVAL_SECONDS:
        action = interactive_window("timer", pdf_id, "")
        if action == "abort":
            raise KeyboardInterrupt("User aborted at checkpoint.")
        elif action == "skip":
            TIMER_START = time.time()
            return "skip"
        else:
            TIMER_START = time.time()
            return "continue"
    return "continue"


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        logger.debug(f"PdfReader failed for {pdf_path}: {e}")
        return ""
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
            parts.append(t)
        except Exception:
            parts.append("")
    return "\n".join(parts).strip()


def df_to_table_text(df: pd.DataFrame, max_chars=3000):
    try:
        if df is None or df.empty:
            return ""
        df = df.astype(str)
        header = "\t".join(df.columns.tolist())
        rows = [header]
        total = len(header)
        for r in df.itertuples(index=False):
            row = "\t".join(str(x) for x in r)
            rows.append(row)
            total += len(row)
            if total > max_chars:
                break
        out = "\n".join(rows)
        return out[:max_chars] + "..." if len(out) > max_chars else out
    except Exception:
        return ""


def extract_tables_pdfplumber(pdf_path: Path):
    out = []
    if pdfplumber is None:
        return out
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for t in tables:
                    try:
                        df = pd.DataFrame(t)
                        if df.shape[0] > 1:
                            df.columns = df.iloc[0].tolist()
                            df = df.iloc[1:].reset_index(drop=True)
                        out.append({"page": i + 1, "df": df})
                    except Exception:
                        continue
    except Exception:
        pass
    return out


def extract_tables_camelot(pdf_path: Path):
    out = []
    if camelot is None:
        return out
    try:
        tables = camelot.read_pdf(str(pdf_path), flavor="stream", pages="all")
        for t in tables:
            try:
                out.append({"page": int(t.page), "df": t.df})
            except Exception:
                continue
    except Exception:
        pass
    return out


def extract_tables_tabula(pdf_path: Path):
    out = []
    if tabula is None:
        return out
    try:
        dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)
        for df in dfs:
            out.append({"page": None, "df": df})
    except Exception:
        pass
    return out


def ocr_pdf_pages(pdf_path: Path):
    if pdfplumber is None or pytesseract is None:
        return []
    texts = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                try:
                    img = page.to_image(resolution=200).original
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    txt = pytesseract.image_to_string(img)
                    texts.append(txt or "")
                except Exception:
                    texts.append("")
    except Exception:
        pass
    return texts


def extract_tables_and_text_for_pdf(pdf_path: Path):
    page_texts = []
    table_blocks = []
    table_dfs = []

    if pdfplumber:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    try:
                        page_texts.append(page.extract_text() or "")
                    except Exception:
                        page_texts.append("")
        except Exception:
            page_texts = []

    if not any(page_texts):
        txt = extract_text_from_pdf(pdf_path)
        page_texts = [txt] if txt else []

    tables = extract_tables_pdfplumber(pdf_path)
    if not tables:
        tables = extract_tables_camelot(pdf_path)
    if not tables:
        tables = extract_tables_tabula(pdf_path)

    for t in tables:
        df = t.get("df")
        if df is None or df.empty:
            continue
        txt = df_to_table_text(df, max_chars=2500)
        if txt:
            page_label = t.get("page", "?")
            block = f"[Table on page {page_label}]\n{txt}"
            table_blocks.append(block)
            table_dfs.append({"df": df, "page": page_label})

    if not any(page_texts):
        page_texts = ocr_pdf_pages(pdf_path)

    return table_blocks, page_texts, table_dfs


def chunk_text(text: str, max_chars=CHUNK_SIZE):
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + max_chars)
        if end < L:
            slice_ = text[start:end]
            idx = slice_.rfind("\n\n")
            if idx == -1:
                idx = slice_.rfind("\n")
            if idx == -1:
                idx = slice_.rfind(" ")
            if idx > 200:
                end = start + idx
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def run_ollama(prompt: str, model: str = MODEL_NAME, timeout: int = 300):
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "stdout": safe_decode(proc.stdout).strip(),
            "stderr": safe_decode(proc.stderr).strip(),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired as e:
        return {"stdout": "", "stderr": f"timeout: {e}", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": f"exception: {e}", "returncode": -1}


def cache_path(pdf_id: str, chunk_hash: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", pdf_id)
    return CACHE_DIR / f"{safe}{chunk_hash}.json"


def load_cache(pdf_id: str, chunk_hash: str):
    p = cache_path(pdf_id, chunk_hash)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_cache_atomic(pdf_id: str, chunk_hash: str, data):
    p = cache_path(pdf_id, chunk_hash)
    try:
        atomic_write_text(p, json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.debug(f"Failed to save cache {p}: {e}")


PROMPT_TEMPLATE = """You are an expert medical information extraction system.
Return STRICT JSON only (no commentary).
If info is missing, use "unknown".
Return ONLY valid JSON.

Fields:
- typical_age_of_onset
- gender_predominance
- life_expectancy
- presenting_symptoms_signs
- key_investigations_for_confirmation
- clinical_presentation
- manageability
- treatability
- onset_age
- evidence
- confidence
"""


def _cleanup_json_like(s: str) -> str:
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", " ", s)
    s = re.sub(r"^(?:\s*Output:|\s*Answer:)\s*", "", s, flags=re.I)
    s = re.sub(r",\s*(\}|])", r"\1", s)
    return s


def _extract_largest_json_block(text: str):
    obj_matches = re.findall(r"(\{[\s\S]*?\})", text)
    arr_matches = re.findall(r"(\[[\s\S]*?\])", text)
    candidates = obj_matches + arr_matches
    if not candidates:
        if "{" in text and "}" in text:
            first = text.find("{")
            last = text.rfind("}")
            if last > first:
                candidates = [text[first : last + 1]]
        elif "[" in text and "]" in text:
            first = text.find("[")
            last = text.rfind("]")
            if last > first:
                candidates = [text[first : last + 1]]
    if not candidates:
        return None
    return max(candidates, key=len)


def validate_and_coerce(parsed):
    out = {
        "typical_age_of_onset": "unknown",
        "gender_predominance": "unknown",
        "life_expectancy": "unknown",
        "presenting_symptoms_signs": [],
        "key_investigations_for_confirmation": [],
        "clinical_presentation": "unknown",
        "manageability": "unknown",
        "treatability": "unknown",
        "onset_age": "unknown",
        "evidence": [],
        "confidence": {k: 0.0 for k in FIELDS},
    }
    if not isinstance(parsed, dict):
        return out

    def as_str(x):
        if x is None:
            return "unknown"
        if isinstance(x, list):
            return "; ".join(str(i) for i in x) if x else "unknown"
        return str(x)

    for f in [
        "typical_age_of_onset",
        "gender_predominance",
        "life_expectancy",
        "clinical_presentation",
        "manageability",
        "treatability",
        "onset_age",
    ]:
        if f in parsed and parsed[f] not in [None, "", "unknown"]:
            out[f] = as_str(parsed[f])

    for f in ["presenting_symptoms_signs", "key_investigations_for_confirmation"]:
        if f in parsed:
            v = parsed[f]
            if isinstance(v, list):
                out[f] = [str(x) for x in v if x not in [None, ""]][:100]
            elif isinstance(v, str) and v.strip().lower() != "unknown":
                parts = re.split(r"\s*[,;]\s*", v.strip())
                out[f] = [p for p in parts if p][:100]

    if "evidence" in parsed and isinstance(parsed["evidence"], list):
        evid = []
        for e in parsed["evidence"][:3]:
            if isinstance(e, dict):
                field = str(e.get("field", "")).strip()
                snippet = str(e.get("snippet", "")).strip()
                if field and snippet:
                    snippet = (snippet[:197] + "...") if len(snippet) > 200 else snippet
                    evid.append({"field": field, "snippet": snippet})
        out["evidence"] = evid

    if "confidence" in parsed and isinstance(parsed["confidence"], dict):
        for k in out["confidence"].keys():
            raw = parsed["confidence"].get(k)
            try:
                val = float(raw)
            except Exception:
                val = 0.0
            val = 0.0 if math.isnan(val) else val
            out["confidence"][k] = max(0.0, min(1.0, val))
    return out


def call_chunk(pdf_id: str, text_chunk: str):
    global QUOTA_TRIGGER_FILE, TIMER_START

    chunk_hash = sha1(text_chunk)[:16]
    cached = load_cache(pdf_id, chunk_hash)
    if cached is not None:
        return cached

    action = maybe_time_prompt(pdf_id)
    if action == "skip":
        err = {"error": "user_skipped_at_timer_checkpoint"}
        save_cache_atomic(pdf_id, chunk_hash, err)
        return err

    prompt = PROMPT_TEMPLATE + "\n\nText chunk:\n" + text_chunk + "\n"
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        resp = run_ollama(prompt, model=MODEL_NAME)
        stdout = resp.get("stdout", "")
        stderr = resp.get("stderr", "")
        rc = resp.get("returncode", 0)

        if is_quota_error(stderr, stdout, rc):
            if QUOTA_TRIGGER_FILE is None:
                QUOTA_TRIGGER_FILE = pdf_id

            logger.error("Quota/limit issue while processing: %s", pdf_id)
            logger.error("stderr: %s", stderr)
            msg = stderr or stdout[:200]
            action_q = interactive_window("quota", pdf_id, msg)

            TIMER_START = time.time()

            if action_q == "skip":
                err = {
                    "error": "quota_exhausted_skipped_chunk",
                    "stderr": stderr,
                    "raw_output": stdout[:1000],
                    "returncode": rc,
                }
                save_cache_atomic(pdf_id, chunk_hash, err)
                return err
            elif action_q == "abort":
                raise KeyboardInterrupt("User aborted after quota/limit error.")
            else:
                last_err = {
                    "error": "quota_exhausted_retrying",
                    "stderr": stderr,
                    "raw_output": stdout[:1000],
                    "returncode": rc,
                }
                time.sleep(1.0)
                continue

        if rc != 0 and not stdout:
            last_err = {"error": f"model_rc_{rc}", "stderr": stderr}
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
                continue
            else:
                save_cache_atomic(pdf_id, chunk_hash, last_err)
                return last_err

        parsed = None
        if stdout:
            try:
                parsed = json.loads(stdout)
            except Exception:
                block = _extract_largest_json_block(stdout)
                if block:
                    cleaned = _cleanup_json_like(block)
                    try:
                        parsed = json.loads(cleaned)
                    except Exception:
                        try:
                            parsed = json.loads(cleaned.replace("'", '"'))
                        except Exception:
                            parsed = None

        if parsed is None:
            last_err = {
                "error": "json_decode_error",
                "raw_output": stdout[:2000],
                "stderr": stderr,
                "returncode": rc,
            }
            if attempt < MAX_RETRIES:
                corrective = (
                    "\n\nThe previous response was not valid JSON. "
                    "Reply ONLY with the JSON object and nothing else. No explanation."
                )
                prompt = PROMPT_TEMPLATE + corrective + "\n\nText chunk:\n" + text_chunk + "\n"
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
                continue
            else:
                save_cache_atomic(pdf_id, chunk_hash, last_err)
                return last_err

        coerced = validate_and_coerce(parsed)
        if stderr:
            coerced["_debug_stderr"] = stderr
        coerced["_debug_returncode"] = rc
        save_cache_atomic(pdf_id, chunk_hash, coerced)
        return coerced

    fail = last_err or {"error": "failed_after_retries_unknown"}
    save_cache_atomic(pdf_id, chunk_hash, fail)
    return fail


def merge_chunk_outputs(json_list):
    merged = {k: "unknown" for k in FIELDS}
    merged["presenting_symptoms_signs"] = []
    merged["key_investigations_for_confirmation"] = []
    merged["evidence"] = []
    merged["confidence"] = {k: 0.0 for k in FIELDS}

    def choose(old, new):
        if not new or new == "unknown":
            return old
        if old == "unknown":
            return new
        return new if len(str(new)) > len(str(old)) else old

    for obj in json_list:
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            if k in FIELDS:
                if isinstance(v, list):
                    if k in ["presenting_symptoms_signs", "key_investigations_for_confirmation"]:
                        merged[k].extend([str(x) for x in v if x])
                    else:
                        merged[k] = choose(merged[k], v)
                else:
                    merged[k] = choose(merged[k], v)
            elif k == "evidence" and isinstance(v, list):
                merged["evidence"].extend(v)
            elif k == "confidence" and isinstance(v, dict):
                for ck, cv in v.items():
                    try:
                        merged["confidence"][ck] = max(merged["confidence"].get(ck, 0.0), float(cv))
                    except Exception:
                        continue

    merged["presenting_symptoms_signs"] = list(dict.fromkeys(merged["presenting_symptoms_signs"]))[:100]
    merged["key_investigations_for_confirmation"] = list(dict.fromkeys(merged["key_investigations_for_confirmation"]))[:100]
    merged["evidence"] = merged["evidence"][:3]
    return merged


def prefer_table_fields(merged: dict, table_dfs: list):
    for t in table_dfs:
        df = t.get("df")
        if df is None:
            continue
        try:
            for c in df.columns:
                if "age" in str(c).lower():
                    vals = df[c].dropna().astype(str).tolist()
                    if vals:
                        merged["typical_age_of_onset"] = vals[0]
                        break
            s = " ".join(df.astype(str).values.flatten())
            m = re.search(r"(\d+)\s*M(ale)?", s, re.I)
            f = re.search(r"(\d+)\s*F(emale)?", s, re.I)
            if m or f:
                merged["gender_predominance"] = f"{m.group(1) if m else '?'}M : {f.group(1) if f else '?'}F"
        except Exception:
            continue
    return merged


def process_single_pdf(pdf_path: Path, worker_id: int = 0):
    folder_name = pdf_path.parent.name
    pdf_name = pdf_path.name
    pdf_id = f"{folder_name}/{pdf_name}"

    logger.info(f"Starting file: {pdf_id}")
    try:
        table_blocks, page_texts, table_dfs = extract_tables_and_text_for_pdf(pdf_path)
        all_chunks = []
        for tb in table_blocks:
            all_chunks.append(tb)
        for pt in page_texts:
            all_chunks.extend(chunk_text(pt, CHUNK_SIZE))

        if not all_chunks:
            txt = extract_text_from_pdf(pdf_path)
            all_chunks = chunk_text(txt, CHUNK_SIZE)
        if not all_chunks:
            return {"file_name": pdf_id, "error": "no_text_or_tables"}

        chunk_results = []
        total_chunks = len(all_chunks)
        for idx, ch in enumerate(all_chunks, start=1):
            logger.info(f"[{pdf_id}] Processing chunk {idx}/{total_chunks}")
            res = call_chunk(pdf_id, ch)
            chunk_results.append(res)

        merged = merge_chunk_outputs(chunk_results)
        merged = prefer_table_fields(merged, table_dfs)
        merged["file_name"] = pdf_id

        safe_folder = re.sub(r"[^0-9A-Za-z_.-]", "_", folder_name)
        safe_file = re.sub(r"[^0-9A-Za-z_.-]", "_", pdf_name)
        safe_name = f"{safe_folder}{safe_file}"

        trace = {"chunks": chunk_results, "merged": merged}
        trace_path = RESULTS_DIR / f"{safe_name}.json"
        atomic_write_text(trace_path, json.dumps(trace, ensure_ascii=False, indent=2))

        table_index = []
        for i, t in enumerate(table_dfs):
            df = t.get("df")
            page = t.get("page", "?")
            if df is None:
                continue
            csv_name = f"{safe_name}table{i+1}.csv"
            csv_path = RESULTS_DIR / csv_name
            try:
                df.to_csv(csv_path, index=False, encoding="utf-8")
                table_index.append({"table_csv": csv_name, "page": page})
            except Exception as e:
                table_index.append({"table_csv": None, "page": page, "error": str(e)})

        if table_index:
            trace["table_index"] = table_index
            atomic_write_text(trace_path, json.dumps(trace, ensure_ascii=False, indent=2))

        logger.info(f"Finished file: {pdf_id}")
        return merged
    except KeyboardInterrupt:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Exception processing {pdf_id}: {e}\n{tb}")
        return {"file_name": pdf_id, "error": str(e)}


def process_all_pdfs_in_folder(folder: Path):
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        logger.info(f"No PDFs found in folder: {folder}")
        return []

    logger.info(f"Processing folder: {folder} ({len(pdfs)} PDFs)")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_single_pdf, p, i % MAX_WORKERS): p for i, p in enumerate(pdfs)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"PDFs in {folder.name}", unit="pdf"):
            p = futures[fut]
            try:
                res = fut.result()
            except KeyboardInterrupt:
                logger.error("Aborted by user.")
                raise
            except Exception as e:
                res = {"file_name": f"{folder.name}/{p.name}", "error": str(e)}
            results.append(res)
    return results


def discover_pdf_folders(parent: Path):
    folders = []
    for child in sorted(parent.iterdir()):
        if child.is_dir() and any(child.glob("*.pdf")):
            folders.append(child)
    return folders


def main():
    global QUOTA_TRIGGER_FILE, TIMER_START

    TIMER_START = time.time()

    logger.info(f"Starting PDF extraction with model={MODEL_NAME}, workers={MAX_WORKERS}")
    logger.info(f"Parent folder: {PARENT_FOLDER.resolve()}")

    folders = discover_pdf_folders(PARENT_FOLDER)
    if not folders:
        logger.info("No PDF subfolders found. Put PDFs into subfolders under the parent directory.")
        return

    for folder in folders:
        logger.info(f"=== Folder: {folder.name} ===")
        try:
            all_results = process_all_pdfs_in_folder(folder)
        except KeyboardInterrupt:
            logger.error("Stopped by user (quota or checkpoint).")
            break

        if not all_results:
            logger.info(f"No results for folder: {folder.name}")
            continue

        df = pd.DataFrame(all_results)
        if "file_name" not in df.columns:
            df["file_name"] = ""
        if "error" not in df.columns:
            df["error"] = ""

        output_csv = PARENT_FOLDER / f"{folder.name}_{OUTPUT_SUFFIX}"
        csv_text = df.to_csv(index=False, encoding="utf-8")
        atomic_write_text(output_csv, csv_text)
        logger.info(f"Saved CSV for folder '{folder.name}' to: {output_csv}")

    if QUOTA_TRIGGER_FILE:
        logger.error(
            "Ollama quota/rate limit first detected while processing: %s",
            QUOTA_TRIGGER_FILE,
        )


if __name__ == "__main__":
    main()
