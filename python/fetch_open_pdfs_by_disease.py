#!/usr/bin/env python3
import csv
import os
import re
import sys
import time
import logging
from urllib.parse import urlparse, quote, unquote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV = os.path.join(REPO_ROOT, "node", "data", "grouped_diseases.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "direct_pdfs.csv")
FAILED_CSV = os.path.join(os.path.dirname(__file__), "failed_download.csv")
DOWNLOAD_PDFS = True
BASE_PDF_DIR = os.path.join(REPO_ROOT, "pdfs")
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 0.4
MAX_TITLE_LENGTH_FILENAME = 120
USER_AGENT = "pdf-finder/1.0 (+https://example.org)"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def safe_filename(s):
    s = (s or "").strip()
    s = re.sub(r'[\\/*?:"<>|]', "_", s)
    s = re.sub(r"\s+", " ", s)
    return s[:MAX_TITLE_LENGTH_FILENAME]


def parse_input_sections(csv_path):
    entries = []
    current_disease = None
    with open(csv_path, newline="", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if ("," not in line) and re.match(r"^[A-Za-z0-9_\- ]+(\.json)?$", line):
                name = line[:-5] if line.endswith(".json") else line
                current_disease = name.strip()
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                url_candidate = None
                for p in reversed(parts):
                    if re.match(r"^https?://", p, re.I):
                        url_candidate = p
                        break
                if url_candidate:
                    entries.append((current_disease, parts, url_candidate))
    return entries


def get_soup(url):
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None, r
        return BeautifulSoup(r.text, "html.parser"), r
    except Exception as e:
        logging.debug(f"get_soup failed for {url}: {e}")
        return None, None


def find_meta_pdf(soup):
    if not soup:
        return None
    m = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if m and m.get("content"):
        return m["content"].strip()
    link_pdf = soup.find("link", attrs={"type": "application/pdf"})
    if link_pdf and link_pdf.get("href"):
        return link_pdf["href"].strip()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if re.search(r"\.pdf($|\?)", href, re.I):
            return href
    return None


def absolutize(href, base_url):
    if not href:
        return None
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("http://") or href.startswith("https://"):
        try:
            p = urlparse(href)
            path = quote(unquote(p.path), safe="/%")
            query = quote(unquote(p.query), safe="=&%")
            rebuilt = f"{p.scheme}://{p.netloc}{path}"
            if query:
                rebuilt += "?" + query
            return rebuilt
        except Exception:
            return href
    if href.startswith("/"):
        p = urlparse(base_url)
        base = f"{p.scheme}://{p.netloc}"
        joined = base + href
        return absolutize(joined, base_url)
    try:
        return requests.compat.urljoin(base_url, href)
    except Exception:
        return href


def head_is_pdf(url):
    try:
        r = session.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400 or r.headers.get("Content-Type") is None:
            r2 = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            ctype = r2.headers.get("Content-Type", "").lower()
            final = r2.url
            if "pdf" in ctype or final.lower().endswith(".pdf"):
                return True, final, None
            return False, final, f"status:{r2.status_code};ctype:{ctype}"
        ctype = r.headers.get("Content-Type", "").lower()
        final = r.url
        if "pdf" in ctype or final.lower().endswith(".pdf"):
            return True, final, None
        return False, final, f"content-type:{ctype}"
    except Exception as e:
        try:
            r2 = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            ctype = r2.headers.get("Content-Type", "").lower()
            final = r2.url
            if "pdf" in ctype or final.lower().endswith(".pdf"):
                return True, final, None
            return False, final, str(e)
        except Exception as e2:
            logging.debug(f"head_is_pdf final error for {url}: {e2}")
            return False, url, str(e2)


def try_variants_and_check(candidates, base_url=None):
    for c in candidates:
        if not c:
            continue
        c_full = absolutize(c, base_url) if base_url else absolutize(c, c)
        ok, final, _reason = head_is_pdf(c_full)
        if ok:
            return final, None
        time.sleep(0.05)
    return None, None


def wiley_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "onlinelibrary.wiley.com" not in parsed.netloc:
        return patterns

    m = re.search(
        r"/doi/(?:full|abs|pdf|epdf|pdfdirect)?/?(.+?)(?:$|[?#])",
        parsed.path + ("?" + (parsed.query or "")),
    )
    if not m:
        segs = [s for s in parsed.path.split("/") if s]
        doi = segs[-1] if segs else None
    else:
        doi = m.group(1).strip()

    if doi:
        doi = doi.rstrip("/")
        enc = quote(doi, safe="/:.%()")
        patterns.append(f"https://onlinelibrary.wiley.com/doi/epdf/{enc}")
        patterns.append(f"https://onlinelibrary.wiley.com/doi/pdf/{enc}")
    if not url.lower().endswith(".pdf"):
        patterns.append(url + "/epdf")
        patterns.append(url + "/pdf")
    return patterns


def nature_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "nature.com" not in parsed.netloc:
        return patterns
    if not url.endswith(".pdf"):
        patterns.append(url + ".pdf")
        patterns.append(re.sub(r"(/articles/[^/]+)$", r"\1.pdf", url))
        patterns.append(re.sub(r"(/articles/[^/]+)$", r"\1.pdf", url) + "?platform=mobile")
    return patterns


def sciencedirect_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "sciencedirect.com" not in parsed.netloc:
        return patterns
    m = re.search(r"/pii/([^/?#]+)", url)
    if m:
        pii = m.group(1)
        patterns.append(
            f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true"
        )
        patterns.append(f"https://www.sciencedirect.com/science/article/pii/{pii}/pdf")
        patterns.append(url + "/pdf")
    patterns.append(url + "/pdf")
    return patterns


def springer_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "link.springer.com" not in parsed.netloc:
        return patterns
    m = re.search(r"/article/(.+)$", url)
    if m:
        doi = m.group(1)
        enc = quote(doi, safe="")
        patterns.append(f"https://link.springer.com/content/pdf/{enc}.pdf")
        patterns.append(f"https://link.springer.com/content/pdf/{doi}.pdf")
    patterns.append(url + ".pdf")
    return patterns


def mdpi_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "mdpi.com" not in parsed.netloc:
        return patterns
    patterns.append(url + "/pdf")
    if url.endswith("/"):
        patterns.append(url + "pdf")
    return patterns


def tandfonline_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "tandfonline.com" not in parsed.netloc:
        return patterns
    patterns.append(url.replace("/doi/abs/", "/doi/pdf/"))
    patterns.append(url.replace("/doi/abs/", "/doi/full/"))
    patterns.append(url + "/pdf")
    return patterns


def bmj_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "bmj.com" in parsed.netloc or "jmg.bmj.com" in parsed.netloc:
        patterns.append(url + ".full.pdf")
        patterns.append(url + ".full")
        patterns.append(url + "/full.pdf")
    return patterns


def pmc_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "pmc.ncbi.nlm.nih.gov" in parsed.netloc:
        if url.endswith("/"):
            patterns.append(url + "pdf")
        patterns.append(url)
    return patterns


def arxiv_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "arxiv.org" in parsed.netloc:
        patterns.append(url.replace("/abs/", "/pdf/") + ".pdf")
        patterns.append(url.replace("/abs/", "/pdf/"))
    return patterns


def bio_medrxiv_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "biorxiv.org" in parsed.netloc or "medrxiv.org" in parsed.netloc:
        patterns.append(url + ".full.pdf")
        if "/content/" in url:
            patterns.append(url + "/full.pdf")
    return patterns


def cell_patterns(url):
    patterns = []
    parsed = urlparse(url)
    if "cell.com" not in parsed.netloc:
        return patterns

    m = re.search(r"(S[0-9A-Za-z\-\(\)]+)", parsed.path)
    if m:
        pii = m.group(1)
        pii_enc = quote(pii, safe="")
        patterns.append(f"https://www.cell.com/action/showPdf?pii={pii_enc}")
        pii_paren_encoded = pii.replace("(", "%28").replace(")", "%29")
        patterns.append(f"https://www.cell.com/action/showPdf?pii={pii_paren_encoded}")
    if not url.lower().endswith(".pdf"):
        patterns.append(url + "/pdf")
    return patterns


DOMAIN_PATTERN_FUNCS = [
    wiley_patterns,
    nature_patterns,
    sciencedirect_patterns,
    springer_patterns,
    mdpi_patterns,
    tandfonline_patterns,
    bmj_patterns,
    pmc_patterns,
    arxiv_patterns,
    bio_medrxiv_patterns,
    cell_patterns,
]


def find_pdf_for_url(url):
    soup, _resp = get_soup(url)
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    if soup:
        meta_pdf = find_meta_pdf(soup)
        if meta_pdf:
            meta_pdf_abs = absolutize(meta_pdf, url)
            ok, final, _reason = head_is_pdf(meta_pdf_abs)
            if ok:
                return final, "meta/pdf-anchor", None

    for func in DOMAIN_PATTERN_FUNCS:
        try:
            pats = func(url)
        except Exception:
            pats = []
        if pats:
            candidate, _ = try_variants_and_check(pats, base_url=url)
            if candidate:
                return candidate, f"pattern:{func.__name__}", None

    general_candidates = []
    if not url.lower().endswith(".pdf"):
        general_candidates += [url + ".pdf", url + "/pdf", url + "/pdf.pdf", url.replace("/abs/", "/pdf/") + ".pdf"]

    doi_candidate = None
    if soup:
        meta_doi = soup.find("meta", attrs={"name": "citation_doi"}) or soup.find(
            "meta", attrs={"name": "DC.Identifier"}
        )
        if meta_doi and meta_doi.get("content"):
            doi_candidate = meta_doi.get("content").strip()
    if doi_candidate:
        general_candidates.append("https://doi.org/" + doi_candidate)
        general_candidates.append("https://doi.org/" + doi_candidate + ".pdf")
        enc = quote(doi_candidate, safe="")
        parsed = urlparse(url)
        if "springer" in parsed.netloc:
            general_candidates.append(f"https://link.springer.com/content/pdf/{enc}.pdf")

    candidate, _ = try_variants_and_check(general_candidates, base_url=url)
    if candidate:
        return candidate, "general-heuristic", None

    if soup:
        anchors = [ab["href"] for ab in soup.find_all("a", href=True)]
        for a in anchors:
            full = absolutize(a, url)
            ok, final, _reason = head_is_pdf(full)
            if ok:
                return final, "anchor-scan", None

    return None, None, None


def download_pdf(url, dest_path):
    try:
        r = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return False, f"status:{r.status_code}"
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(8192):
                if chunk:
                    fh.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file {INPUT_CSV} not found.")
        sys.exit(1)

    entries = parse_input_sections(INPUT_CSV)
    logging.info(f"Parsed {len(entries)} URL entries from {INPUT_CSV}")

    results = []
    failed_downloads = []

    if DOWNLOAD_PDFS:
        ensure_dir(BASE_PDF_DIR)

    for disease, original_row_parts, url in tqdm(entries, desc="Processing"):
        disease_name = disease or "UNKNOWN"
        safe_disease = safe_filename(disease_name)
        disease_dir = os.path.join(BASE_PDF_DIR, safe_disease)
        if DOWNLOAD_PDFS:
            ensure_dir(disease_dir)

        try:
            pdf_url, method, reason = find_pdf_for_url(url)
        except Exception as e:
            logging.debug(f"find_pdf_for_url error for {url}: {e}")
            pdf_url, method, reason = None, None, str(e)

        if pdf_url:
            title = None
            try:
                soup, _ = get_soup(url)
                if soup:
                    t = soup.find("meta", attrs={"name": "citation_title"}) or soup.find(
                        "meta", property="og:title"
                    ) or soup.find("title")
                    if t:
                        title = t.get("content") if t.get("content") else (t.text or None)
            except Exception:
                title = None
            if not title:
                title = os.path.basename(urlparse(pdf_url).path) or "paper"
            filename = safe_filename(title)
            if not filename.lower().endswith(".pdf"):
                filename = filename + ".pdf"
            dest_path = os.path.join(disease_dir, filename) if DOWNLOAD_PDFS else ""

            i = 1
            base, ext = os.path.splitext(filename)
            while DOWNLOAD_PDFS and os.path.exists(os.path.join(disease_dir, filename)):
                filename = f"{base}_{i}{ext}"
                dest_path = os.path.join(disease_dir, filename)
                i += 1

            if DOWNLOAD_PDFS:
                ok, dl_reason = download_pdf(pdf_url, dest_path)
                if not ok:
                    logging.warning(f"Failed to download confirmed pdf: {pdf_url}  -> {dl_reason}")
                    failed_downloads.append(
                        {"disease": disease_name, "name": filename, "link": pdf_url, "reason": dl_reason}
                    )
                    saved_as = ""
                else:
                    logging.info(f"Downloaded -> {dest_path}")
                    saved_as = dest_path
            else:
                saved_as = ""

            results.append(
                {
                    "disease": disease_name,
                    "input_row": ",".join(original_row_parts),
                    "input_url": url,
                    "pdf_url": pdf_url,
                    "method": method or "",
                    "saved_as": saved_as,
                }
            )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["disease", "input_row", "input_url", "pdf_url", "method", "saved_as"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    with open(FAILED_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["disease", "name", "link", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fd in failed_downloads:
            writer.writerow(fd)

    logging.info(f"Done. Confirmed PDFs: {len(results)}. Output: {OUTPUT_CSV}")
    logging.info(f"Failed downloads: {len(failed_downloads)}. Output: {FAILED_CSV}")
    if DOWNLOAD_PDFS:
        logging.info(f"PDF folders under: {BASE_PDF_DIR}")


if __name__ == "__main__":
    main()
