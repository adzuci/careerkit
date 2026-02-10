#!/usr/bin/env python3
"""
Automated Job Search for Liz Blackwell
Senior Data & Analytics Roles (Remote, $150k+)

Criteria:
- Compensation: $150,000+ base salary
- Location: Fully remote, United States
- Company types: B2B SaaS, healthcare/life sciences, data platforms, analytics consulting
- Exclude: crypto-only, on-site/hybrid, junior roles

Target Roles:
- Leadership: Director/Head/Manager of Analytics, Analytics Engineering
- IC: Principal Analytics Engineer, Staff Data Engineer, Data Scientist

Core Skills:
- dbt, Snowflake, Databricks
- SQL, Python
- Analytics Engineering, Modern BI
- Healthcare/life sciences experience (bonus)
"""

import pandas as pd
from jobspy import scrape_jobs
import warnings
warnings.filterwarnings('ignore')


def contains_any(text: str, keywords: list[str]) -> bool:
    """Return True if any keyword appears in the given text (case-insensitive)."""
    if text is None:
        return False
    text = text.lower()
    return any(k in text for k in keywords)


def score_role(text: str, title: str) -> int:
    """
    Score a role based on fit for Liz's profile.
    Higher score = better fit.
    """
    if text is None:
        text = ""
    if title is None:
        title = ""
    
    text = text.lower()
    title = title.lower()
    combined = text + " " + title
    
    score = 0
    
    # Core tech stack (high weight)
    tech_keywords = ["dbt", "snowflake", "databricks", "sql", "python", "data warehouse"]
    for k in tech_keywords:
        if k in combined:
            score += 3
    
    # Analytics focus (high weight)
    analytics_keywords = ["analytics engineering", "analytics engineer", "data platform", 
                         "modern data stack", "bi ", "business intelligence", "data governance"]
    for k in analytics_keywords:
        if k in combined:
            score += 3
    
    # Leadership indicators (medium weight)
    leadership_keywords = ["director", "head of", "manager", "lead", "principal", "staff", 
                          "senior", "stakeholder", "client delivery", "team"]
    for k in leadership_keywords:
        if k in combined:
            score += 2
    
    # Healthcare/life sciences bonus
    health_keywords = ["healthcare", "health care", "life sciences", "pharma", "biotech", 
                      "medical", "clinical"]
    for k in health_keywords:
        if k in combined:
            score += 2
    
    # Remote indicator
    if "remote" in combined:
        score += 1
    
    # Negative signals
    bad_keywords = ["junior", "entry level", "intern", "crypto", "blockchain", "web3",
                   "hybrid only", "on-site only", "clearance required"]
    for k in bad_keywords:
        if k in combined:
            score -= 5
    
    return score


# === Search Configuration for Liz Blackwell ===

SEARCH_TERMS = [
    "Director of Analytics",
    "Head of Analytics",
    "Analytics Engineering Manager",
    "Principal Analytics Engineer",
    "Director Data Platform",
    "Staff Data Engineer",
    "Senior Analytics Engineer",
    "Principal Data Engineer",
    "Analytics Manager",
    "Head of Data Platforms",
]

SITES = [
    "indeed",
    "linkedin",
    "glassdoor",
]

LOCATION = "United States"
RESULTS_PER_SITE = 50
HOURS_OLD = 336  # 14 days

# Keywords that indicate good fit
GOOD_KEYWORDS = [
    "dbt",
    "snowflake",
    "databricks",
    "analytics engineering",
    "data platform",
    "remote",
    "healthcare",
    "life sciences",
    "sql",
    "python",
]

# Keywords that indicate poor fit
BAD_KEYWORDS = [
    "crypto",
    "blockchain",
    "web3",
    "on-site only",
    "junior",
    "entry level",
    "intern",
    "clearance",
]

print("=" * 60)
print("Job Search for Liz Blackwell")
print("Senior Data & Analytics Roles (Remote, $150k+)")
print("=" * 60)
print()

# Collect all jobs
all_jobs = []

for site in SITES:
    for term in SEARCH_TERMS:
        print(f"Scraping {site} for '{term}'...")
        try:
            jobs_df = scrape_jobs(
                site_name=site,
                search_term=term,
                location=LOCATION,
                results_wanted=RESULTS_PER_SITE,
                hours_old=HOURS_OLD,
                country_indeed="USA",
                is_remote=True,  # Filter for remote roles
            )
            if len(jobs_df) > 0:
                jobs_df["site"] = site
                jobs_df["search_term"] = term
                all_jobs.append(jobs_df)
                print(f"  -> Retrieved {len(jobs_df)} results.")
            else:
                print(f"  -> No results.")
        except Exception as e:
            print(f"  !! Error: {e}")

if not all_jobs:
    print("\nNo jobs retrieved. Creating empty output file.")
    empty_df = pd.DataFrame(columns=[
        "company", "title", "salary_range", "why_fits", "url", "location", "site"
    ])
    empty_df.to_csv("liz_jobs.csv", index=False)
    exit(0)

raw_df = pd.concat(all_jobs, ignore_index=True)
print(f"\nTotal raw jobs collected: {len(raw_df)}")

# Create combined text for filtering
df = raw_df.copy()

# Ensure required columns exist
for col in ["description", "title", "company", "location"]:
    if col not in df.columns:
        df[col] = ""

df["description"] = df["description"].fillna("")
df["title"] = df["title"].fillna("")
df["company"] = df["company"].fillna("")
df["location"] = df["location"].fillna("")

df["search_text"] = (
    df["title"].astype(str) + " " + 
    df["description"].astype(str) + " " + 
    df["company"].astype(str)
).str.lower()

# Filter out bad matches
bad_mask = df["search_text"].apply(lambda t: contains_any(t, BAD_KEYWORDS))
df_filtered = df[~bad_mask].copy()
print(f"After removing bad keywords: {len(df_filtered)} roles")

# Score remaining roles
df_filtered["fit_score"] = df_filtered.apply(
    lambda row: score_role(row["search_text"], row["title"]), axis=1
)

# Sort by fit score
df_filtered = df_filtered.sort_values("fit_score", ascending=False)

# Deduplicate
dedupe_cols = ["title", "company"]
df_unique = df_filtered.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
print(f"After deduplication: {len(df_unique)} unique roles")

# Filter for high-fit roles (score >= 10)
df_high_fit = df_unique[df_unique["fit_score"] >= 10].copy()
print(f"High-fit roles (score >= 10): {len(df_high_fit)}")

# Take top 30 for manual review
df_top = df_high_fit.head(30).copy()

# Generate "why fits" summary
def generate_why_fits(row):
    reasons = []
    text = str(row.get("search_text", "")).lower()
    title = str(row.get("title", "")).lower()
    
    if any(k in text for k in ["dbt", "snowflake", "databricks"]):
        reasons.append("Modern data stack (dbt/Snowflake/Databricks)")
    if any(k in text for k in ["analytics engineer", "analytics engineering"]):
        reasons.append("Analytics engineering focus")
    if any(k in title for k in ["director", "head", "manager", "principal", "staff"]):
        reasons.append("Senior/leadership level")
    if any(k in text for k in ["healthcare", "health care", "life sciences"]):
        reasons.append("Healthcare/life sciences domain")
    if any(k in text for k in ["remote"]):
        reasons.append("Remote position")
    if any(k in text for k in ["stakeholder", "client"]):
        reasons.append("Client/stakeholder facing")
    
    return "; ".join(reasons) if reasons else "General analytics/data role"

df_top["why_fits"] = df_top.apply(generate_why_fits, axis=1)

# Extract salary info if available
def get_salary_range(row):
    for col in ["min_amount", "max_amount", "salary", "compensation"]:
        if col in row and pd.notna(row.get(col)):
            return str(row.get(col))
    return "Not specified"

if "min_amount" in df_top.columns or "max_amount" in df_top.columns:
    df_top["salary_range"] = df_top.apply(
        lambda r: f"${r.get('min_amount', 'N/A'):,.0f} - ${r.get('max_amount', 'N/A'):,.0f}" 
        if pd.notna(r.get('min_amount')) else "Not specified",
        axis=1
    )
else:
    df_top["salary_range"] = "Not specified"

# Get URL column
url_col = None
for col in ["job_url", "job_url_direct", "url", "link"]:
    if col in df_top.columns:
        url_col = col
        break

if url_col:
    df_top["application_link"] = df_top[url_col]
else:
    df_top["application_link"] = ""

# Select output columns
output_cols = ["company", "title", "salary_range", "why_fits", "application_link", 
               "location", "site", "fit_score"]
output_cols = [c for c in output_cols if c in df_top.columns]

df_output = df_top[output_cols].copy()

# Save results
df_output.to_csv("liz_jobs.csv", index=False)
print(f"\n{'=' * 60}")
print(f"Saved {len(df_output)} high-fit roles to: liz_jobs.csv")
print(f"{'=' * 60}")

# Display preview
print("\nTop 10 Results Preview:")
print("-" * 60)
for i, row in df_output.head(10).iterrows():
    print(f"\n{i+1}. {row.get('title', 'N/A')} @ {row.get('company', 'N/A')}")
    print(f"   Location: {row.get('location', 'N/A')}")
    print(f"   Salary: {row.get('salary_range', 'N/A')}")
    print(f"   Why fits: {row.get('why_fits', 'N/A')}")
    print(f"   Score: {row.get('fit_score', 0)}")
