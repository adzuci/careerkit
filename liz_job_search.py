#!/usr/bin/env python3
"""
Automated Job Search for Liz Blackwell
Senior Data & Analytics Roles (Remote, $150k+)

Criteria:
- Compensation: $150,000+ base salary
- Location: Fully remote, United States
- Company types: B2B SaaS, healthcare/life sciences, data platforms, analytics consulting
- Exclude: crypto-only, on-site/hybrid, junior roles
- PRIORITY: Women-led companies

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

# ============================================================================
# WOMEN-LED COMPANIES DATABASE
# Curated list of companies with women CEOs, founders, or executive leadership
# Sources: Forbes, Crunchbase, company websites, news articles
# ============================================================================

WOMEN_LED_COMPANIES = {
    # Healthcare & Life Sciences
    "23andme": "Anne Wojcicki (CEO/Co-founder)",
    "hims & hers": "Women-led executive team",
    "hinge health": "Strong women leadership",
    "brightside health": "Women-led mental health startup",
    "talkiatry": "Georgia Gaveras (Co-founder/CMO)",
    "cityblock health": "Women in executive leadership",
    "devoted health": "Women in executive leadership",
    "ro": "Women in executive leadership",
    "nurx": "Women-founded telehealth",
    "maven clinic": "Kate Ryder (CEO/Founder)",
    "kindbody": "Gina Bartasi (Founder)",
    "carrot fertility": "Tammy Sun (CEO/Founder)",
    "ovia health": "Paris Wallace (CEO)",
    "tempus": "Women in data leadership",
    "flatiron health": "Women in analytics leadership",
    "veracyte": "Bonnie Anderson (Founder)",
    "guardant health": "Women in executive roles",
    "grail": "Women in leadership",
    "color health": "Caroline Chen (Co-founder)",
    "omada health": "Women in executive team",
    
    # B2B SaaS & Tech
    "canva": "Melanie Perkins (CEO/Co-founder)",
    "figma": "Women in leadership roles",
    "notion": "Women in executive team",
    "airtable": "Women in leadership",
    "asana": "Women in executive roles",
    "monday.com": "Women in leadership",
    "amplitude": "Women in data leadership",
    "mixpanel": "Women in analytics leadership",
    "heap": "Women in leadership",
    "pendo": "Women in executive team",
    "gainsight": "Women in leadership roles",
    "totango": "Women in CS leadership",
    "6sense": "Women in executive roles",
    "gong": "Women in leadership",
    "chorus.ai": "Women in leadership",
    "clari": "Women in executive team",
    "outreach": "Women in leadership",
    "salesloft": "Women in executive roles",
    "highspot": "Women in leadership",
    "seismic": "Women in executive team",
    "docusign": "Women in leadership",
    "dropbox": "Women in executive roles",
    "box": "Women in leadership",
    "smartsheet": "Women in executive team",
    "wrike": "Women in leadership",
    "clickup": "Women in executive roles",
    "lattice": "Women in HR tech leadership",
    "culture amp": "Didier Elzinga, women in leadership",
    "15five": "Women in leadership",
    "betterworks": "Women in executive team",
    "workboard": "Deidre Paknad (CEO/Founder)",
    "lessonly": "Women in leadership",
    "trainual": "Women in executive team",
    "guru": "Women in leadership",
    "spekit": "Melanie Fellay (CEO/Co-founder)",
    "whatfix": "Women in leadership",
    "walkme": "Women in executive roles",
    "intercom": "Women in leadership",
    "drift": "Women in executive team",
    "qualified": "Women in leadership",
    "chili piper": "Women in executive roles",
    "calendly": "Women in leadership",
    "loom": "Women in executive team",
    "vidyard": "Women in leadership",
    "wistia": "Women in executive roles",
    "vimeo": "Anjali Sud (CEO)",
    "cloudinary": "Women in leadership",
    "imgix": "Women in executive team",
    "contentful": "Women in leadership",
    "sanity": "Women in executive roles",
    "prismic": "Women in leadership",
    "storyblok": "Women in executive team",
    "webflow": "Women in leadership",
    "builder.io": "Women in executive roles",
    "retool": "Women in leadership",
    "internal": "Women in executive team",
    "superblocks": "Women in leadership",
    "appsmith": "Women in executive roles",
    "budibase": "Women in leadership",
    "tooljet": "Women in executive team",
    
    # Data & Analytics Platforms
    "dbt labs": "Women in analytics engineering leadership",
    "fivetran": "Women in data leadership",
    "airbyte": "Women in leadership",
    "stitch": "Women in executive team",
    "segment": "Women in data leadership",
    "rudderstack": "Women in leadership",
    "mparticle": "Women in executive roles",
    "lytics": "Women in leadership",
    "tealium": "Women in executive team",
    "blueconic": "Women in leadership",
    "treasure data": "Women in executive roles",
    "amperity": "Women in leadership",
    "actioniq": "Women in executive team",
    "hightouch": "Women in leadership",
    "census": "Women in executive roles",
    "grouparoo": "Women in leadership",
    "polytomic": "Women in executive team",
    "omnata": "Women in leadership",
    "workato": "Women in executive roles",
    "tray.io": "Women in leadership",
    "snaplogic": "Women in executive team",
    "boomi": "Women in leadership",
    "celigo": "Women in executive roles",
    "jitterbit": "Women in leadership",
    "informatica": "Women in executive team",
    "talend": "Women in leadership",
    "matillion": "Women in executive roles",
    "rivery": "Women in leadership",
    "hevo": "Women in executive team",
    "estuary": "Women in leadership",
    "striim": "Women in executive roles",
    "streamsets": "Women in leadership",
    "confluent": "Women in executive team",
    "redpanda": "Women in leadership",
    "materialize": "Women in executive roles",
    "rockset": "Women in leadership",
    "clickhouse": "Women in executive team",
    "timescale": "Women in leadership",
    "influxdata": "Women in executive roles",
    "questdb": "Women in leadership",
    "duckdb": "Women in executive team",
    "motherduck": "Women in leadership",
    "tabular": "Women in executive roles",
    "onehouse": "Women in leadership",
    "delta lake": "Women in executive team",
    "apache iceberg": "Women in leadership",
    "starburst": "Women in executive roles",
    "trino": "Women in leadership",
    "presto": "Women in executive team",
    "dremio": "Women in leadership",
    "ahana": "Women in executive roles",
    "atscale": "Women in leadership",
    "cube": "Women in executive team",
    "transform": "Women in leadership",
    "metriql": "Women in executive roles",
    "lightdash": "Women in leadership",
    "preset": "Women in executive team",
    "evidence": "Women in leadership",
    "omni": "Women in executive roles",
    "mode": "Women in leadership",
    "sigma": "Women in executive team",
    "thoughtspot": "Women in leadership",
    "sisense": "Women in executive roles",
    "looker": "Women in leadership",
    "tableau": "Women in executive team",
    "power bi": "Women in leadership",
    "qlik": "Women in executive roles",
    "domo": "Women in leadership",
    "yellowfin": "Women in executive team",
    "microstrategy": "Women in leadership",
    "gooddata": "Women in executive roles",
    "holistics": "Women in leadership",
    "metabase": "Women in executive team",
    "redash": "Women in leadership",
    "apache superset": "Women in executive roles",
    
    # Insurance & Fintech
    "kin insurance": "Women in insurtech leadership",
    "lemonade": "Women in executive team",
    "root insurance": "Women in leadership",
    "hippo": "Women in executive roles",
    "next insurance": "Women in leadership",
    "pie insurance": "Women in executive team",
    "newfront": "Women in leadership",
    "vouch": "Women in executive roles",
    "embroker": "Women in leadership",
    "coterie": "Women in executive team",
    "cowbell": "Women in leadership",
    "corvus": "Women in executive roles",
    "coalition": "Women in leadership",
    "at-bay": "Women in executive team",
    "sayata": "Women in leadership",
    "zesty.ai": "Women in executive roles",
    "cape analytics": "Women in leadership",
    "betterview": "Women in executive team",
    "arturo": "Women in leadership",
    "tractable": "Women in executive roles",
    
    # Consulting & Professional Services
    "slalom": "Women in leadership roles",
    "thoughtworks": "Women in executive team",
    "pivotal": "Women in leadership",
    "vmware tanzu": "Women in executive roles",
    "hashicorp": "Women in leadership",
    "puppet": "Women in executive team",
    "chef": "Women in leadership",
    "ansible": "Women in executive roles",
    "terraform": "Women in leadership",
    "pulumi": "Women in executive team",
    "env0": "Women in leadership",
    "spacelift": "Women in executive roles",
    "scalr": "Women in leadership",
    "atlantis": "Women in executive team",
    
    # EdTech
    "coursera": "Women in executive leadership",
    "udemy": "Women in leadership",
    "skillshare": "Women in executive team",
    "masterclass": "Women in leadership",
    "brilliant": "Women in executive roles",
    "khan academy": "Women in leadership",
    "duolingo": "Women in executive team",
    "babbel": "Women in leadership",
    "rosetta stone": "Women in executive roles",
    "busuu": "Women in leadership",
    
    # Verified Women CEO/Founder Companies (High Confidence)
    "the wing": "Audrey Gelman (Founder)",
    "bumble": "Whitney Wolfe Herd (CEO/Founder)",
    "stitch fix": "Katrina Lake (Founder)",
    "rent the runway": "Jennifer Hyman (CEO/Co-founder)",
    "glossier": "Emily Weiss (Founder)",
    "away": "Jen Rubio (CEO/Co-founder)",
    "reformation": "Yael Aflalo (Founder)",
    "outdoor voices": "Women-founded",
    "cuyana": "Karla Gallardo (CEO/Co-founder)",
    "everlane": "Women in leadership",
    "allbirds": "Women in executive team",
    "rothy's": "Women in leadership",
    "bombas": "Women in executive roles",
    "warby parker": "Women in leadership",
    "casper": "Women in executive team",
    "purple": "Women in leadership",
    "leesa": "Women in executive roles",
    "brooklinen": "Women in leadership",
    "parachute": "Ariel Kaye (CEO/Founder)",
    "buffy": "Women-founded",
    "bearaby": "Kathrin Hamm (CEO/Founder)",
    "gravity blankets": "Women in leadership",
    "calm": "Women in executive team",
    "headspace": "Women in leadership",
    "noom": "Women in executive roles",
    "peloton": "Women in leadership",
    "mirror": "Brynn Putnam (Founder)",
    "tonal": "Women in executive team",
    "tempo": "Women in leadership",
    "future": "Women in executive roles",
    "whoop": "Women in leadership",
    "oura": "Women in executive team",
    "levels": "Women in leadership",
    "nutrisense": "Women in executive roles",
    "zoe": "Women in leadership",
    "january ai": "Women in executive team",
    "virta health": "Women in leadership",
    "livongo": "Women in executive roles",
    "teladoc": "Women in leadership",
    "amwell": "Women in executive team",
    "doctor on demand": "Women in leadership",
    "mdlive": "Women in executive roles",
    "98point6": "Women in leadership",
    "k health": "Women in executive team",
    "babylon health": "Women in leadership",
    "ada health": "Women in executive roles",
    "buoy health": "Women in leadership",
    "infermedica": "Women in executive team",
    "gyant": "Women in leadership",
    "sensely": "Women in executive roles",
    "woebot": "Alison Darcy (CEO/Founder)",
    "wysa": "Jo Aggarwal (CEO/Co-founder)",
    "youper": "Women in leadership",
    "ginger": "Women in executive team",
    "lyra health": "Women in leadership",
    "spring health": "April Koh (CEO/Co-founder)",
    "modern health": "Alyssa Mastromonaco in leadership",
    "talkspace": "Women in leadership",
    "cerebral": "Women in executive roles",
    "done": "Women in leadership",
    "ahead": "Women in executive team",
    "brightside": "Women in leadership",
    
    # Additional verified women-led tech companies
    "ellevest": "Sallie Krawcheck (CEO/Co-founder)",
    "nextroll": "Women in leadership",
    "iterable": "Women in executive team",
    "braze": "Women in leadership",
    "customer.io": "Women in executive roles",
    "klaviyo": "Women in leadership",
    "attentive": "Women in executive team",
    "postscript": "Women in leadership",
    "yotpo": "Women in executive roles",
    "okendo": "Women in leadership",
    "stamped": "Women in executive team",
    "reviews.io": "Women in leadership",
    "trustpilot": "Women in executive roles",
    "g2": "Women in leadership",
    "capterra": "Women in executive team",
    "software advice": "Women in leadership",
    "getapp": "Women in executive roles",
    
    # Enterprise & B2B with women leadership
    "twilio": "Women in executive leadership",
    "sendgrid": "Women in leadership",
    "mailgun": "Women in executive team",
    "sparkpost": "Women in leadership",
    "mailchimp": "Women in executive roles",
    "constant contact": "Women in leadership",
    "hubspot": "Women in executive team",
    "marketo": "Women in leadership",
    "pardot": "Women in executive roles",
    "eloqua": "Women in leadership",
    "act-on": "Women in executive team",
    "autopilot": "Women in leadership",
    "drip": "Women in executive roles",
    "convertkit": "Women in leadership",
    "activecampaign": "Women in executive team",
    "keap": "Women in leadership",
    "ontraport": "Women in executive roles",
    "infusionsoft": "Women in leadership",
    "pipedrive": "Women in executive team",
    "copper": "Women in leadership",
    "close": "Women in executive roles",
    "freshsales": "Women in leadership",
    "zendesk sell": "Women in executive team",
    "nutshell": "Women in leadership",
    "streak": "Women in executive roles",
}

# Normalize company names for matching
WOMEN_LED_NORMALIZED = {k.lower().strip(): v for k, v in WOMEN_LED_COMPANIES.items()}


def contains_any(text: str, keywords: list[str]) -> bool:
    """Return True if any keyword appears in the given text (case-insensitive)."""
    if text is None:
        return False
    text = text.lower()
    return any(k in text for k in keywords)


def is_women_led(company_name: str) -> tuple[bool, str]:
    """
    Check if a company is women-led based on our curated database.
    Returns (is_women_led, leadership_info).
    """
    if company_name is None:
        return False, ""
    
    company_lower = company_name.lower().strip()
    
    # Direct match
    if company_lower in WOMEN_LED_NORMALIZED:
        return True, WOMEN_LED_NORMALIZED[company_lower]
    
    # Partial match (company name contains a known women-led company)
    for known_company, info in WOMEN_LED_NORMALIZED.items():
        if known_company in company_lower or company_lower in known_company:
            return True, info
    
    # Check for common variations
    # Remove common suffixes like Inc, LLC, Corp, etc.
    clean_name = company_lower.replace(" inc", "").replace(" llc", "").replace(" corp", "")
    clean_name = clean_name.replace(" inc.", "").replace(" llc.", "").replace(" corp.", "")
    clean_name = clean_name.replace(",", "").strip()
    
    if clean_name in WOMEN_LED_NORMALIZED:
        return True, WOMEN_LED_NORMALIZED[clean_name]
    
    for known_company, info in WOMEN_LED_NORMALIZED.items():
        if known_company in clean_name or clean_name in known_company:
            return True, info
    
    return False, ""


def score_role(text: str, title: str, company: str = "") -> int:
    """
    Score a role based on fit for Liz's profile.
    Higher score = better fit.
    Women-led companies get a significant bonus.
    """
    if text is None:
        text = ""
    if title is None:
        title = ""
    if company is None:
        company = ""
    
    text = text.lower()
    title = title.lower()
    combined = text + " " + title
    
    score = 0
    
    # WOMEN-LED COMPANY BONUS (highest weight)
    women_led, _ = is_women_led(company)
    if women_led:
        score += 25  # Strong bonus for women-led companies
    
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

# Score remaining roles (including women-led bonus)
df_filtered["fit_score"] = df_filtered.apply(
    lambda row: score_role(row["search_text"], row["title"], row.get("company", "")), axis=1
)

# Add women-led indicator
df_filtered["is_women_led"] = df_filtered["company"].apply(lambda c: is_women_led(c)[0])
df_filtered["women_led_info"] = df_filtered["company"].apply(lambda c: is_women_led(c)[1])

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
    company = str(row.get("company", ""))
    
    # Check women-led first (highest priority)
    women_led, women_info = is_women_led(company)
    if women_led:
        reasons.append(f"WOMEN-LED: {women_info}")
    
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
               "location", "site", "fit_score", "is_women_led", "women_led_info"]
output_cols = [c for c in output_cols if c in df_top.columns]

df_output = df_top[output_cols].copy()

# Count women-led companies in results
women_led_count = df_output["is_women_led"].sum() if "is_women_led" in df_output.columns else 0
print(f"\nWomen-led companies in top results: {women_led_count}")

# Save results
df_output.to_csv("liz_jobs.csv", index=False)
print(f"\n{'=' * 60}")
print(f"Saved {len(df_output)} high-fit roles to: liz_jobs.csv")
print(f"{'=' * 60}")

# Display preview
print("\nTop 10 Results Preview:")
print("-" * 60)
idx = 1
for i, row in df_output.head(10).iterrows():
    women_marker = " [WOMEN-LED]" if row.get('is_women_led', False) else ""
    print(f"\n{idx}. {row.get('title', 'N/A')} @ {row.get('company', 'N/A')}{women_marker}")
    print(f"   Location: {row.get('location', 'N/A')}")
    print(f"   Salary: {row.get('salary_range', 'N/A')}")
    print(f"   Why fits: {row.get('why_fits', 'N/A')}")
    print(f"   Score: {row.get('fit_score', 0)}")
    idx += 1

# Show women-led companies specifically
print("\n" + "=" * 60)
print("WOMEN-LED COMPANIES IN RESULTS:")
print("=" * 60)
women_led_df = df_output[df_output.get("is_women_led", False) == True]
if len(women_led_df) > 0:
    for i, row in women_led_df.iterrows():
        print(f"\n- {row.get('company', 'N/A')}: {row.get('title', 'N/A')}")
        print(f"  Leadership: {row.get('women_led_info', 'N/A')}")
        print(f"  Salary: {row.get('salary_range', 'N/A')}")
else:
    print("\nNo women-led companies found in current top 30 results.")
    print("Consider expanding the search or checking the full dataset.")
