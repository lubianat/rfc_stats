from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

BASE = Path(__file__).resolve().parent
PARENT = BASE.parent
OUT_PATH = PARENT / "ngff_rfcs.yaml"

RAW_GITHUB_BASE = "https://raw.githubusercontent.com/ome/ngff/refs/heads/main/"
WEB_BASE = "https://ngff.openmicroscopy.org/"
API_BASE = "https://api.github.com/repos/ome/ngff/"
GITHUB_PR = "https://github.com/ome/ngff/pull/"

REQUEST_TIMEOUT = 30

# Fields that, once present in the existing ngff_rfcs.yaml, are treated as
# manually curated and are NOT overwritten by a fresh scrape. Everything else
# is recomputed on every run.
PRESERVED_FIELDS = ("title", "description", "url", "id", "pr_url")

KNOWN_FIELDS = {"role", "name", "github_handle", "institution", "date", "status"}
DEFAULT_HEADERS = ["role", "name", "github_handle", "institution", "date", "status"]

# A status link points at a sub-record: ./reviews/1/index , ./comments/2/index , etc.
SUBLINK_RE = re.compile(
    r"\((?:\./)?(reviews|comments|responses|versions)/([0-9a-zA-Z]+)/index\)"
)
# OME-Zarr version mentions like NGFF 0.5 / OME-Zarr 0.5 / version 0.5
OZ_VERSION_RE = re.compile(r"\b(?:OME-Zarr|NGFF|version)\s*0\.\d+\b", re.IGNORECASE)

# State codes defined by RFC-1's process diagram. D=Draft, R=Review, S=Spec/adopted.
# R9 is the "closed/withdrawn" terminal state. These come from the `## Status`
# section of each index.md and are the machine-readable substitute for the free
# text in the (to-be-retired) listing.csv.
STATE_CODE_RE = re.compile(r"\b([DRS])([0-9])\b")
STATE_LABELS = {
    "D": "Draft",
    "R": "Under review",
    "S": "Adopted",
}


def state_label(code: str | None) -> str | None:
    if not code:
        return None
    letter = code[0].upper()
    if code.upper() == "R9":
        return "Withdrawn"
    if letter == "S" and code != "S0":
        # S1-S4 are increasing degrees of adoption; S4 is fully adopted.
        return "Adopted"
    return STATE_LABELS.get(letter)


# ----------------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------------
def _session() -> requests.Session:
    s = requests.Session()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    s.headers["Accept"] = "application/vnd.github+json"
    return s


SESSION = _session()


def fetch_text(url: str) -> str | None:
    try:
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        logging.warning("Failed to fetch %s: %s", url, exc)
        return None
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.text


def fetch_json(url: str) -> Any | None:
    try:
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        logging.warning("Failed to fetch %s: %s", url, exc)
        return None
    if r.status_code in (403, 404):
        logging.warning(
            "GitHub API %s -> %s (rate limit / not found)", url, r.status_code
        )
        return None
    r.raise_for_status()
    return r.json()


# ----------------------------------------------------------------------------
# Markdown text helpers (carried over from contribs scraper)
# ----------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_markdown(text: str) -> str:
    value = text.strip()
    if not value:
        return ""
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    value = normalize_whitespace(value)
    return "" if value == "-" else value


def extract_links(text: str) -> list[tuple[str, str]]:
    """Return [(text, url), ...] for every markdown link in the cell."""
    return re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)


def normalize_header(header: str) -> str:
    n = strip_markdown(header).lower()
    n = normalize_whitespace(re.sub(r"[^a-z0-9 ]+", " ", n))
    if n == "role":
        return "role"
    if n == "name":
        return "name"
    if "github" in n:
        return "github_handle"
    if "institution" in n:
        return "institution"
    if n == "date":
        return "date"
    if n == "status":
        return "status"
    return ""


def looks_like_header_row(headers: list[str]) -> bool:
    recognized = {h for h in headers if h in KNOWN_FIELDS}
    return "name" in recognized and len(recognized) >= 2


def normalize_row_length(cells: list[str], headers: list[str]) -> list[str]:
    if len(cells) < len(headers):
        return cells + [""] * (len(headers) - len(cells))
    if len(cells) == len(headers):
        return cells
    return cells[: len(headers) - 1] + [" ".join(cells[len(headers) - 1 :])]


def cells_to_record(cells: list[str], headers: list[str]) -> dict[str, str]:
    """Keep both the cleaned text and the raw markdown (raw_* keys keep links)."""
    norm = normalize_row_length(cells, headers)
    record: dict[str, str] = {}
    for header, cell in zip(headers, norm):
        if header in KNOWN_FIELDS:
            record[header] = strip_markdown(cell)
            record[f"raw_{header}"] = cell.strip()
    return record


# ----------------------------------------------------------------------------
# list-table parsing
# ----------------------------------------------------------------------------
def parse_list_table_block(block: str) -> list[dict[str, str]]:
    rows: list[list[str]] = []
    current: list[str] | None = None
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith(":"):
            continue
        m = re.match(r"^\s*\*\s*-\s*(.*)$", line)
        if m:
            if current:
                rows.append(current)
            current = [m.group(1).strip()]
            continue
        m = re.match(r"^\s*-\s*(.*)$", line)
        if m and current is not None:
            current.append(m.group(1).strip())
            continue
        if current:
            current[-1] = normalize_whitespace(f"{current[-1]} {line.strip()}".strip())
    if current:
        rows.append(current)
    if not rows:
        return []
    candidate = [normalize_header(c) for c in rows[0]]
    if looks_like_header_row(candidate):
        headers, data_rows = candidate, rows[1:]
    else:
        headers, data_rows = DEFAULT_HEADERS[: len(rows[0])], rows
    return [cells_to_record(r, headers) for r in data_rows if any(r)]


def parse_list_tables(md: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for m in re.finditer(r"```{list-table}[^\n]*\n(?P<body>.*?)```", md, re.DOTALL):
        records.extend(parse_list_table_block(m.group("body")))
    return records


# ----------------------------------------------------------------------------
# pipe-table parsing
# ----------------------------------------------------------------------------
def split_md_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_md_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$", line))


def parse_markdown_tables(md: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    lines = md.splitlines()
    i, in_fence = 0, False
    while i < len(lines):
        cur = lines[i]
        if cur.strip().startswith("```"):
            in_fence = not in_fence
            i += 1
            continue
        if in_fence or i + 1 >= len(lines):
            i += 1
            continue
        if "|" not in cur or not is_md_separator(lines[i + 1]):
            i += 1
            continue
        headers = [normalize_header(c) for c in split_md_row(cur)]
        recognized = {h for h in headers if h in KNOWN_FIELDS}
        if "name" not in recognized or len(recognized) < 3:
            i += 1
            continue
        i += 2
        while i < len(lines):
            row = lines[i]
            if not row.strip() or "|" not in row or row.strip().startswith("```"):
                break
            cells = split_md_row(row)
            if cells and any(c.strip() for c in cells):
                records.append(cells_to_record(cells, headers))
            i += 1
    return records


def deduplicate(records: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    out = []
    fields = DEFAULT_HEADERS
    for rec in records:
        key = tuple(rec.get(f, "") for f in fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# Domain helpers
# ----------------------------------------------------------------------------
def split_people(value: str) -> list[str]:
    return [p.strip() for p in re.split(r",|;", value) if p.strip()]


def people_list(value: str) -> list[dict[str, str]]:
    return [
        {"name": strip_markdown(n)} for n in split_people(value) if strip_markdown(n)
    ]


def parse_date(value: str) -> str | None:
    value = strip_markdown(value)
    m = re.search(r"\d{4}-\d{2}-\d{2}", value)
    return m.group(0) if m else None


def sub_id_sort_key(sub_id: str) -> tuple[int, str]:
    """'1' -> (1,''), '1b' -> (1,'b') so 1b sorts right after 1."""
    m = re.match(r"^(\d+)([a-zA-Z]*)$", sub_id)
    if m:
        return int(m.group(1)), m.group(2)
    return 10**9, sub_id


def base_sub_id(sub_id: str) -> str | None:
    """'1b' -> '1' (its related base). Returns None for a base id."""
    m = re.match(r"^(\d+)([a-zA-Z]+)$", sub_id)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------
# Per-RFC builder
# ----------------------------------------------------------------------------
def parse_records(md: str) -> list[dict[str, str]]:
    records = parse_list_tables(md)
    records.extend(parse_markdown_tables(md))
    return deduplicate(records)


def extract_title_description(
    md: str, rfc_number: int
) -> tuple[str | None, str | None]:
    title = None
    m = re.search(rf"^#\s*RFC-{rfc_number}:\s*(.+)$", md, re.MULTILINE)
    if m:
        title = strip_markdown(m.group(1))
    description = None
    m = re.search(r"^##\s*Overview\s*\n+(.*?)(?:\n#|\Z)", md, re.DOTALL | re.MULTILINE)
    if m:
        first_para = m.group(1).split("\n\n", 1)[0].strip()
        # drop HTML comment blocks
        first_para = re.sub(r"<!--.*?-->", "", first_para, flags=re.DOTALL).strip()
        first_para = first_para.split("\n\n", 1)[0].strip()
        if first_para:
            description = strip_markdown(first_para)
    return title, description


def extract_status(md: str) -> dict[str, Any]:
    """Pull the state code + human label + prose from the `## Status` section.

    This replaces the Status/Date columns previously hand-maintained in
    listing.csv. Returns {state_code, status, status_text}. RFC-7 etc. with an
    empty Status section yield None values (equivalent to listing.csv 'TBD').
    """
    m = re.search(
        r"^##\s*Status\s*\n+(.*?)(?:\n```|\n\|)", md, re.DOTALL | re.MULTILINE
    )
    text = ""
    if m:
        # take prose only, up to the start of the Record table
        text = re.sub(r"<!--.*?-->", "", m.group(1), flags=re.DOTALL)
        text = normalize_whitespace(strip_markdown(text))
    code = None
    cm = STATE_CODE_RE.search(text)
    if cm:
        code = f"{cm.group(1)}{cm.group(2)}"
    return {
        "state_code": code,
        "status": state_label(code),
        "status_text": text or None,
    }


def extract_year(md: str, rfc: dict[str, Any]) -> int | None:
    """Best-effort single year for the RFC (the listing.csv 'Date' column).

    Prefer the PR merge year, else the earliest author date, else any 4-digit
    year mentioned in the Status prose.
    """
    if rfc.get("pr_merge_date"):
        return int(rfc["pr_merge_date"][:4])
    dates = [
        r["date"]
        for key in ("reviews", "comments", "responses", "versions")
        for r in rfc.get(key, [])
        if r.get("date")
    ]
    if rfc.get("authors_dates"):
        dates.extend(rfc["authors_dates"])
    if dates:
        return int(min(dates)[:4])
    ym = re.search(r"\b(20\d{2})\b", rfc.get("status_text") or "")
    return int(ym.group(1)) if ym else None


def extract_oz_versions(md: str) -> list[str]:
    versions = set()
    for m in OZ_VERSION_RE.finditer(md):
        num = re.search(r"0\.\d+", m.group(0))
        if num:
            versions.add(num.group(0))
    return sorted(versions)


def find_pr_number(md: str, records: list[dict[str, str]]) -> str | None:
    # PR links appear in author status cells and elsewhere
    haystack = (
        md
        + " "
        + " ".join(r.get(f"raw_{k}", "") for r in records for k in ("status", "name"))
    )
    m = re.search(r"ome/ngff/pull/(\d+)", haystack)
    return m.group(1) if m else None


def pr_number_from_url(pr_url: str | None) -> str | None:
    """Extract the trailing PR number from a pull-request URL, or None."""
    if not pr_url:
        return None
    m = re.search(r"/pull/(\d+)", pr_url)
    return m.group(1) if m else None


def build_sub_records(
    records: list[dict[str, str]], rfc_number: int, kind: str
) -> list[dict[str, Any]]:
    """Build review/comment/response/version entries keyed off status-column links."""
    by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        status_raw = rec.get("raw_status", "")
        for link_kind, sub_id in SUBLINK_RE.findall(status_raw):
            if link_kind != kind:
                continue
            entry = by_id.setdefault(
                sub_id,
                {
                    "id": f"rfc/{rfc_number}/{kind}/{sub_id}",
                    "url": f"{WEB_BASE}rfc/{rfc_number}/{kind}/{sub_id}",
                    "authors": [],
                    "date": None,
                },
            )
            # Merge author names from this record's name cell
            for person in people_list(rec.get("raw_name", rec.get("name", ""))):
                if person not in entry["authors"]:
                    entry["authors"].append(person)
            date = parse_date(rec.get("date", ""))
            if date and not entry["date"]:
                entry["date"] = date
            # A "b" id means a follow-up tied to its base (e.g. 1b -> 1)
            base = base_sub_id(sub_id)
            if base:
                entry["related_to"] = f"rfc/{rfc_number}/{kind}/{base}"
    return [by_id[k] for k in sorted(by_id, key=sub_id_sort_key)]


def role_of(rec: dict[str, str]) -> str:
    """Determine a row's role.

    Most RFC tables have an explicit Role column. RFC-8 (and the commented
    template rows) instead omit Role and encode it in the Status column
    (every row's Status reads "Author"/"Endorser"/...). Fall back to the
    Status text when there is no Role column so those rows aren't dropped.
    """
    role = strip_markdown(rec.get("role", "")).lower()
    if role:
        return role
    return strip_markdown(rec.get("status", "")).lower()


def build_rfc(
    rfc_number: int, existing: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    url = RAW_GITHUB_BASE + f"rfc/{rfc_number}/index.md"
    md = fetch_text(url)
    if md is None:
        logging.info("RFC %s: no index.md", rfc_number)
        return None

    records = parse_records(md)
    title, description = extract_title_description(md, rfc_number)

    authors: list[dict[str, str]] = []
    endorsers: list[dict[str, str]] = []
    author_dates: list[str] = []
    for rec in records:
        role = role_of(rec)
        people = people_list(rec.get("raw_name", rec.get("name", "")))
        if "author" in role:
            for p in people:
                if p not in authors:
                    authors.append(p)
            d = parse_date(rec.get("date", ""))
            if d:
                author_dates.append(d)
        elif "endorser" in role:
            for p in people:
                if p not in endorsers:
                    endorsers.append(p)

    pr_number = find_pr_number(md, records)

    status_info = extract_status(md)

    rfc: dict[str, Any] = {
        "id": f"rfc/{rfc_number}",
        "url": f"{WEB_BASE}rfc/{rfc_number}",
        "title": title,
        "description": description,
        "status": status_info["status"],
        "state_code": status_info["state_code"],
        "status_text": status_info["status_text"],
        "year": None,  # filled after sub-records/PR date are known
        "authors": authors,
        "endorsers": endorsers,
        "oz_versions": extract_oz_versions(md),
        "pr_url": f"{GITHUB_PR}{pr_number}" if pr_number else None,
        "pr_merge_date": None,  # filled in below if API reachable
        "last_text_commit_date": None,  # filled in below if API reachable
        "reviews": build_sub_records(records, rfc_number, "reviews"),
        "comments": build_sub_records(records, rfc_number, "comments"),
        "responses": build_sub_records(records, rfc_number, "responses"),
        "versions": build_sub_records(records, rfc_number, "versions"),
    }

    # Apply manually-curated fields (title/description/url/id/pr_url) BEFORE the
    # GitHub enrichment so that a hand-edited pr_url drives the pr_merge_date
    # lookup. If the curated entry supplies a pr_url, its PR number wins over the
    # one scraped from the index.md.
    apply_preserved_fields(rfc, existing)
    pr_number = pr_number_from_url(rfc.get("pr_url")) or pr_number

    enrich_with_github(rfc, rfc_number, pr_number)

    rfc["authors_dates"] = author_dates
    rfc["year"] = extract_year(md, rfc)
    del rfc["authors_dates"]  # internal scratch field, not part of the schema

    counts = {k: len(rfc[k]) for k in ("reviews", "comments", "responses", "versions")}
    logging.info(
        "RFC %s: authors=%d endorsers=%d %s",
        rfc_number,
        len(authors),
        len(endorsers),
        ", ".join(f"{k}={v}" for k, v in counts.items()),
    )
    return rfc


def enrich_with_github(
    rfc: dict[str, Any], rfc_number: int, pr_number: str | None
) -> None:
    """Best-effort GitHub API enrichment. Skips silently on rate-limit/no-token."""
    # Last commit touching the RFC's index.md
    commits = fetch_json(
        API_BASE + f"commits?path=rfc/{rfc_number}/index.md&per_page=1"
    )
    if isinstance(commits, list) and commits:
        date = commits[0].get("commit", {}).get("committer", {}).get("date")
        if date:
            rfc["last_text_commit_date"] = date[:10]

    if pr_number:
        pr = fetch_json(API_BASE + f"pulls/{pr_number}")
        if isinstance(pr, dict):
            merged = pr.get("merged_at")
            if merged:
                rfc["pr_merge_date"] = merged[:10]


# ----------------------------------------------------------------------------
# Preserve manually-curated fields across runs
# ----------------------------------------------------------------------------
def load_existing() -> dict[str, Any]:
    """Return the existing {rfc<n>: {...}} map from OUT_PATH, or {} if absent.

    Tolerant of a missing or unparseable file: in either case we just return an
    empty map and the run proceeds as a fresh scrape.
    """
    if not OUT_PATH.exists():
        return {}
    try:
        loaded = yaml.safe_load(OUT_PATH.read_text()) or {}
    except yaml.YAMLError as exc:
        logging.warning("Could not parse existing %s (%s); ignoring", OUT_PATH, exc)
        return {}
    rfcs = loaded.get("ngff_rfcs")
    return rfcs if isinstance(rfcs, dict) else {}


def apply_preserved_fields(
    fresh: dict[str, Any], existing: dict[str, Any] | None
) -> None:
    """Overlay manually-curated fields from `existing` onto a freshly-built RFC.

    A preserved field is only carried over when it is actually present in the
    existing entry (i.e. the key exists and the value is not None/empty). This
    lets a hand-edited title or description survive a re-scrape while still
    allowing newly-scraped values to fill fields that were never set.
    """
    if not isinstance(existing, dict):
        return
    for field in PRESERVED_FIELDS:
        if field not in existing:
            continue
        value = existing[field]
        if value is None or value == "":
            continue
        if fresh.get(field) != value:
            logging.info(
                "RFC %s: preserving curated %s (keeping existing value)",
                fresh.get("id"),
                field,
            )
        fresh[field] = value


# ----------------------------------------------------------------------------
# Discovery + main
# ----------------------------------------------------------------------------
def discover_rfc_numbers(max_probe: int = 60) -> list[int]:
    """Prefer the git tree API; fall back to sequential probing of raw files."""
    tree = fetch_json(API_BASE + "git/trees/main?recursive=1")
    if isinstance(tree, dict) and tree.get("tree"):
        numbers = set()
        for item in tree["tree"]:
            m = re.match(r"^rfc/(\d+)/index\.md$", item.get("path", ""))
            if m:
                numbers.add(int(m.group(1)))
        if numbers:
            logging.info("Discovered %d RFCs via git tree", len(numbers))
            return sorted(numbers)

    logging.info("Falling back to probing raw files for RFC numbers")
    found = []
    for n in range(1, max_probe + 1):
        head_url = RAW_GITHUB_BASE + f"rfc/{n}/index.md"
        try:
            r = SESSION.head(head_url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        except requests.RequestException:
            continue
        if r.status_code == 200:
            found.append(n)
    return found


def main() -> None:
    rfc_numbers = discover_rfc_numbers()
    if not rfc_numbers:
        logging.error("No RFCs discovered; aborting")
        return

    existing = load_existing()
    if existing:
        logging.info(
            "Loaded %d existing RFC entries for field preservation", len(existing)
        )

    rfcs: dict[str, Any] = {}
    for n in rfc_numbers:
        rfc = build_rfc(n, existing.get(f"rfc{n}"))
        if rfc:
            rfcs[f"rfc{n}"] = rfc

    data = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "https://github.com/ome/ngff",
        "ngff_rfcs": dict(sorted(rfcs.items(), key=lambda kv: int(kv[0][3:]))),
    }
    OUT_PATH.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True))
    logging.info("Wrote %s with %d RFC entries", OUT_PATH, len(rfcs))


if __name__ == "__main__":
    main()
