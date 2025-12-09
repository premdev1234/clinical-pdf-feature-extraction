#!/usr/bin/env python3
import os
import json
import math
import random
import hashlib
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from extract_clinical_features import (
    REPO_ROOT,
    RESULTS_DIR,
    CACHE_DIR,
    MODEL_NAME,
    CHUNK_SIZE,
    extract_tables_and_text_for_pdf,
    chunk_text,
    merge_chunk_outputs,
    prefer_table_fields,
    extract_text_from_pdf,
    run_ollama,
    _cleanup_json_like,
    _extract_largest_json_block,
    validate_and_coerce,
)

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cache_path_ui(pdf_id: str, chunk_hash: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", pdf_id)
    return CACHE_DIR / f"ui_{safe}{chunk_hash}.json"


def load_cache_ui(pdf_id: str, chunk_hash: str):
    p = cache_path_ui(pdf_id, chunk_hash)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_cache_ui(pdf_id: str, chunk_hash: str, data):
    p = cache_path_ui(pdf_id, chunk_hash)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5


def call_chunk_ui(pdf_id: str, text_chunk: str):
    chunk_hash = sha1(text_chunk)[:16]
    cached = load_cache_ui(pdf_id, chunk_hash)
    if cached is not None:
        return cached

    prompt = (
        "You are an expert medical information extraction system.\n"
        "Return STRICT JSON only (no commentary).\n"
        "If info is missing, use \"unknown\".\n"
        "Return ONLY valid JSON.\n\n"
        "Fields:\n"
        "- typical_age_of_onset\n"
        "- gender_predominance\n"
        "- life_expectancy\n"
        "- presenting_symptoms_signs\n"
        "- key_investigations_for_confirmation\n"
        "- clinical_presentation\n"
        "- manageability\n"
        "- treatability\n"
        "- onset_age\n"
        "- evidence\n"
        "- confidence\n\n"
        "Text chunk:\n"
        + text_chunk
        + "\n"
    )

    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        resp = run_ollama(prompt, model=MODEL_NAME)
        stdout = resp.get("stdout", "")
        stderr = resp.get("stderr", "")
        rc = resp.get("returncode", 0)

        if rc != 0 and not stdout:
            last_err = {"error": f"model_rc_{rc}", "stderr": stderr}
            if attempt < MAX_RETRIES:
                st.info(f"[{pdf_id}] retrying (rc={rc}) attempt {attempt+1}/{MAX_RETRIES}")
                import time
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
                continue
            else:
                save_cache_ui(pdf_id, chunk_hash, last_err)
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
                prompt = prompt + corrective
                import time
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)) + random.random())
                continue
            else:
                save_cache_ui(pdf_id, chunk_hash, last_err)
                return last_err

        coerced = validate_and_coerce(parsed)
        if stderr:
            coerced["_debug_stderr"] = stderr
        coerced["_debug_returncode"] = rc
        save_cache_ui(pdf_id, chunk_hash, coerced)
        return coerced

    fail = last_err or {"error": "failed_after_retries_unknown"}
    save_cache_ui(pdf_id, chunk_hash, fail)
    return fail


def process_single_pdf_bytes(name: str, data: bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    pdf_id = f"uploaded/{name}"
    try:
        table_blocks, page_texts, table_dfs = extract_tables_and_text_for_pdf(tmp_path)
        all_chunks = []

        for tb in table_blocks:
            all_chunks.append(tb)
        for pt in page_texts:
            all_chunks.extend(chunk_text(pt, CHUNK_SIZE))

        if not all_chunks:
            txt = extract_text_from_pdf(tmp_path)
            all_chunks = chunk_text(txt, CHUNK_SIZE)
        if not all_chunks:
            return {"file_name": name, "error": "no_text_or_tables"}

        chunk_results = []
        total_chunks = len(all_chunks)
        progress_text = st.empty()
        progress_bar = st.progress(0)

        for idx, ch in enumerate(all_chunks, start=1):
            progress_text.text(f"{name}: processing chunk {idx}/{total_chunks}")
            res = call_chunk_ui(pdf_id, ch)
            chunk_results.append(res)
            progress_bar.progress(idx / total_chunks)

        progress_bar.empty()
        progress_text.empty()

        merged = merge_chunk_outputs(chunk_results)
        merged = prefer_table_fields(merged, table_dfs)
        merged["file_name"] = name
        return merged
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def main():
    st.set_page_config(
        page_title="SCA Clinical Feature Extraction",
        layout="wide",
    )

    st.title("Clinical Feature Extraction from PDFs")
    st.markdown(
        """
        Upload up to **10 PDF files** (e.g. SCA clinical papers).
        The app will:
        1. Extract text and tables from each PDF.
        2. Call an Ollama model (`gpt-oss:120b-cloud` by default).
        3. Produce a structured table of clinical features per PDF.
        """
    )

    st.info(
        "Ensure the Ollama daemon is running and the model "
        f"`{MODEL_NAME}` is available (see `ollama ls`).",
        icon="ℹ️",
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload at most 10 PDFs.",
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("You uploaded more than 10 files; only the first 10 will be processed.")
            uploaded_files = uploaded_files[:10]

        if st.button("Run Extraction"):
            rows = []
            for f in uploaded_files:
                st.subheader(f"Processing: {f.name}")
                row = process_single_pdf_bytes(f.name, f.getvalue())
                rows.append(row)

            df = pd.DataFrame(rows)
            st.subheader("Extracted Clinical Features")
            st.dataframe(df, use_container_width=True)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="uploaded_pdfs_clinical_data.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

