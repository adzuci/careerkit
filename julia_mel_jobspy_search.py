#!/usr/bin/env python3
"""Find high-signal MEL/research/evaluation roles for Julia with JobSpy.

This script intentionally uses keyword heuristics instead of pretending to make
final application decisions. Treat the output as a first-pass lead queue, then
verify each posting on the employer site before applying.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SEARCH_TERMS = [
    "monitoring evaluation learning manager",
    "research evaluation manager nonprofit",
    "learning evaluation philanthropy",
    "impact measurement manager",
    "senior researcher evaluation",
    "program evaluation manager remote",
    "research director social impact",
    "evaluation consultant philanthropy",
]

HIGH_SIGNAL_TERMS = [
    "evaluation",
    "monitoring",
    "learning",
    "mel",
    "research",
    "impact",
    "social impact",
    "philanthropy",
    "foundation",
    "nonprofit",
    "non-profit",
    "ngo",
    "qualitative",
    "quantitative",
    "policy",
    "stata",
    "nvivo",
    "excel",
    "dashboard",
    "data platform",
    "client",
    "stakeholder",
    "grantmaking",
    "theory of change",
]

PREFERRED_ORG_TERMS = [
    "foundation",
    "philanthropy",
    "nonprofit",
    "non-profit",
    "ngo",
    "social impact",
    "academic",
    "university",
    "research institute",
    "public health",
    "policy",
    "education",
]

SENIOR_SCOPE_TERMS = [
    "senior",
    "manager",
    "director",
    "lead",
    "principal",
    "consultant",
    "head of",
]

BLOCKED_TERMS = [
    "intern",
    "internship",
    "entry level",
    "junior",
    "assistant",
    "coordinator",
    "claims examiner",
    "insurance adjuster",
    "underwriter",
    "clinical trial",
    "nurse",
    "sales development",
    "business development representative",
]

BAD_COMPANIES = [
    "carefirst",
    "bluecross",
    "blue cross",
    "bcbs",
]


@dataclass(frozen=True)
class SearchConfig:
    sites: list[str]
    location: str
    results_wanted: int
    hours_old: int
    min_salary: int
    include_near_miss_nonprofit: bool
    output_csv: Path
    output_md: Path
    cv_path: Path | None


def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(
        description="Scrape and rank Julia-fit MEL/research/evaluation roles."
    )
    parser.add_argument(
        "--site",
        dest="sites",
        action="append",
        choices=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
        help="JobSpy site to search. Repeat for multiple sites.",
    )
    parser.add_argument("--location", default="United States")
    parser.add_argument("--results-wanted", type=int, default=25)
    parser.add_argument("--hours-old", type=int, default=168)
    parser.add_argument("--min-salary", type=int, default=120_000)
    parser.add_argument(
        "--strict-salary",
        action="store_true",
        help="Exclude nonprofit near-misses that only top out near the salary floor.",
    )
    parser.add_argument(
        "--cv-path",
        type=Path,
        default=None,
        help='Optional local CV path, e.g. "/Users/adam/Downloads/J. Kern CV GitLab.pdf".',
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("julia_mel_jobspy_results.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("julia_mel_jobspy_results.md"),
    )
    args = parser.parse_args()

    return SearchConfig(
        sites=args.sites or ["indeed", "linkedin", "zip_recruiter", "google"],
        location=args.location,
        results_wanted=args.results_wanted,
        hours_old=args.hours_old,
        min_salary=args.min_salary,
        include_near_miss_nonprofit=not args.strict_salary,
        output_csv=args.output_csv,
        output_md=args.output_md,
        cv_path=args.cv_path,
    )


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(safe_text(item) for item in value)
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (ImportError, TypeError, ValueError):
        pass
    return str(value)


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def joined_text(row: pd.Series) -> str:
    fields = [
        row.get("title"),
        row.get("company"),
        row.get("location"),
        row.get("description"),
        row.get("company_industry"),
        row.get("skills"),
    ]
    return compact_whitespace(" ".join(safe_text(field) for field in fields).lower())


def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term.lower() in text for term in terms)


def keyword_hits(text: str, terms: Iterable[str]) -> list[str]:
    return sorted({term for term in terms if term.lower() in text})


def salary_floor(row: pd.Series) -> float | None:
    min_amount = row.get("min_amount")
    max_amount = row.get("max_amount")
    values = []
    for value in [min_amount, max_amount]:
        if value is None:
            continue
        try:
            import pandas as pd

            if pd.isna(value):
                continue
        except ImportError:
            pass
        values.append(value)
    if not values:
        return None
    return float(max(values))


def salary_passes(row: pd.Series, config: SearchConfig, text: str) -> bool:
    max_salary = salary_floor(row)
    if max_salary is None:
        return False
    if max_salary >= config.min_salary:
        return True
    if not config.include_near_miss_nonprofit:
        return False
    return max_salary >= config.min_salary * 0.85 and contains_any(
        text, PREFERRED_ORG_TERMS
    )


def is_remote_or_boston(row: pd.Series) -> bool:
    location = safe_text(row.get("location")).lower()
    description = safe_text(row.get("description")).lower()
    is_remote = bool(row.get("is_remote"))
    return is_remote or "remote" in location or "remote" in description or "boston" in location


def score_row(row: pd.Series, config: SearchConfig) -> tuple[int, list[str]]:
    text = joined_text(row)
    hits = keyword_hits(text, HIGH_SIGNAL_TERMS)
    senior_hits = keyword_hits(text, SENIOR_SCOPE_TERMS)
    org_hits = keyword_hits(text, PREFERRED_ORG_TERMS)

    score = 0
    score += len(hits) * 2
    score += len(senior_hits) * 3
    score += len(org_hits) * 2

    title = safe_text(row.get("title")).lower()
    if any(term in title for term in ["evaluation", "research", "learning", "mel"]):
        score += 8
    if any(term in title for term in ["manager", "director", "senior", "principal"]):
        score += 6
    if is_remote_or_boston(row):
        score += 5
    if salary_floor(row) and salary_floor(row) >= config.min_salary:
        score += 5

    reasons = []
    if hits:
        reasons.append("keywords: " + ", ".join(hits[:8]))
    if senior_hits:
        reasons.append("senior scope: " + ", ".join(senior_hits[:5]))
    if org_hits:
        reasons.append("sector: " + ", ".join(org_hits[:5]))
    return score, reasons


def read_cv_keywords(cv_path: Path | None) -> list[str]:
    """Best-effort CV extraction for local runs; avoids failing if deps are absent."""
    if cv_path is None:
        return []
    if not cv_path.exists():
        print(f"CV path not found, skipping CV extraction: {cv_path}")
        return []
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        print("Install pypdf to extract CV text: python3 -m pip install pypdf")
        return []

    reader = PdfReader(str(cv_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).lower()
    return keyword_hits(text, HIGH_SIGNAL_TERMS + PREFERRED_ORG_TERMS)


def scrape_all(config: SearchConfig) -> pd.DataFrame:
    try:
        import pandas as pd
        from jobspy import scrape_jobs
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install JobSpy and pandas with "
            "`python3 -m pip install python-jobspy pandas`."
        ) from exc

    frames: list[pd.DataFrame] = []
    for term in DEFAULT_SEARCH_TERMS:
        print(f"Searching {config.sites}: {term}")
        try:
            jobs = scrape_jobs(
                site_name=config.sites,
                search_term=term,
                location=config.location,
                results_wanted=config.results_wanted,
                hours_old=config.hours_old,
                is_remote=True,
                linkedin_fetch_description=True,
            )
        except Exception as exc:  # Job boards can rate-limit or change markup.
            print(f"Search failed for {term!r}: {exc}")
            continue
        if not jobs.empty:
            jobs["search_term"] = term
            frames.append(jobs)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["job_url", "title", "company"], keep="first"
    )


def filter_and_rank(jobs: pd.DataFrame, config: SearchConfig) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for _, row in jobs.iterrows():
        text = joined_text(row)
        if contains_any(text, BLOCKED_TERMS):
            continue
        if contains_any(text, BAD_COMPANIES):
            continue
        if not is_remote_or_boston(row):
            continue
        if not contains_any(text, HIGH_SIGNAL_TERMS):
            continue
        if not contains_any(text, SENIOR_SCOPE_TERMS):
            continue
        if not salary_passes(row, config, text):
            continue

        score, reasons = score_row(row, config)
        updated = row.to_dict()
        updated["fit_score"] = score
        updated["fit_reasons"] = "; ".join(reasons)
        updated["salary_max_or_floor"] = salary_floor(row)
        rows.append(updated)

    if not rows:
        return pd.DataFrame()

    ranked = pd.DataFrame(rows).sort_values(
        by=["fit_score", "salary_max_or_floor"], ascending=[False, False]
    )
    preferred_columns = [
        "fit_score",
        "title",
        "company",
        "location",
        "min_amount",
        "max_amount",
        "currency",
        "is_remote",
        "date_posted",
        "site",
        "job_url",
        "job_url_direct",
        "fit_reasons",
        "description",
    ]
    return ranked[[column for column in preferred_columns if column in ranked.columns]]


def write_markdown(ranked: pd.DataFrame, config: SearchConfig, cv_hits: list[str]) -> None:
    import pandas as pd

    lines = [
        "# Julia MEL JobSpy results",
        "",
        "Generated by `julia_mel_jobspy_search.py`.",
        "",
        "## Search settings",
        "",
        f"- Sites: {', '.join(config.sites)}",
        f"- Location: {config.location}",
        f"- Hours old: {config.hours_old}",
        f"- Salary floor: ${config.min_salary:,}",
        f"- Nonprofit near-misses included: {config.include_near_miss_nonprofit}",
    ]
    if cv_hits:
        lines.append(f"- CV keyword hits used for calibration: {', '.join(cv_hits[:20])}")
    lines.extend(
        [
            "",
            "## Top leads",
            "",
            "| Score | Company | Title | Salary | Location | Why it matched | Link |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for _, row in ranked.head(25).iterrows():
        salary = ""
        if not pd.isna(row.get("min_amount")) or not pd.isna(row.get("max_amount")):
            salary = (
                f"{safe_text(row.get('currency') or 'USD')} "
                f"{safe_text(row.get('min_amount'))}-{safe_text(row.get('max_amount'))}"
            )
        link = row.get("job_url_direct") or row.get("job_url") or ""
        lines.append(
            "| {score} | {company} | {title} | {salary} | {location} | {reasons} | {link} |".format(
                score=safe_text(row.get("fit_score")),
                company=safe_text(row.get("company")).replace("|", "/"),
                title=safe_text(row.get("title")).replace("|", "/"),
                salary=salary.replace("|", "/"),
                location=safe_text(row.get("location")).replace("|", "/"),
                reasons=safe_text(row.get("fit_reasons")).replace("|", "/"),
                link=link,
            )
        )

    lines.extend(
        [
            "",
            "## Human review checklist",
            "",
            "1. Open the employer application link and confirm the role is still accepting applications.",
            "2. Reject aggregator-only leads that do not resolve to an employer posting.",
            "3. Confirm remote US or Boston-hybrid status and travel expectations.",
            "4. Confirm the salary band with a recruiter before investing tailoring time.",
            "5. Add feedback labels such as `good`, `bad_company`, `too_junior`, `salary_low`, `not_remote`, or `applied`.",
        ]
    )
    config.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = parse_args()
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install pandas and JobSpy with "
            "`python3 -m pip install python-jobspy pandas`."
        ) from exc

    cv_hits = read_cv_keywords(config.cv_path)
    jobs = scrape_all(config)
    if jobs.empty:
        raise SystemExit("No jobs returned. Try fewer sites or a broader time window.")

    ranked = filter_and_rank(jobs, config)
    if ranked.empty:
        raise SystemExit("No jobs passed filters. Try --strict-salary off or broaden search terms.")

    ranked.to_csv(config.output_csv, index=False)
    write_markdown(ranked, config, cv_hits)
    print(f"Wrote {len(ranked)} ranked roles to {config.output_csv}")
    print(f"Wrote Markdown summary to {config.output_md}")


if __name__ == "__main__":
    main()
