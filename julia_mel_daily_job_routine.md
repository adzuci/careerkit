# Julia MEL daily job-search routine

This routine pairs `julia_mel_jobspy_search.py` with a scheduled Claude task.
It is designed to produce a short daily email, collect feedback, and improve
the next search without storing Julia's CV in the repo.

## Local daily command

Run from the repo root:

```bash
python3 -m pip install python-jobspy pandas pypdf
python3 julia_mel_jobspy_search.py \
  --cv-path "/Users/adam/Downloads/J. Kern CV GitLab.pdf" \
  --output-csv julia_mel_jobspy_results.csv \
  --output-md julia_mel_jobspy_results.md
```

Notes:

- Use `python3` instead of `python` if that is how your local environment exposes
  Python.
- `pypdf` is optional, but lets the script extract keyword hints from the CV.
- The CV stays local. Do not commit it.
- Job boards can rate-limit or change markup; treat failures as expected
  operational noise and rerun with fewer sites if needed.
- The script blocks BCBS/CareFirst by default.

## Claude scheduled task proposal

Create a daily scheduled Claude routine with this prompt:

> Every weekday morning, run Julia's MEL job search routine from the careerkit
> repo. Use `/Users/adam/Downloads/J. Kern CV GitLab.pdf` only as local context;
> do not attach it to email and do not commit it. Run
> `python3 julia_mel_jobspy_search.py --cv-path "/Users/adam/Downloads/J. Kern CV GitLab.pdf"`.
> Review `julia_mel_jobspy_results.md` and send me a concise email with only
> high-signal roles. Exclude BCBS/CareFirst, on-site roles, junior roles, roles
> below $120k unless they are exceptional nonprofit/philanthropy fits, and
> roles that do not clearly connect to MEL, research, evaluation, impact
> measurement, analytics leadership, data platforms, or client delivery.
> Include a feedback section where I can reply with labels:
> `good`, `apply`, `bad_company`, `too_junior`, `salary_low`, `not_remote`,
> `wrong_sector`, `already_seen`, or `more_like_this: <role/company>`.

## Daily email format

Subject:

```text
Julia MEL roles: <N> strong leads for <YYYY-MM-DD>
```

Body:

```text
Top leads
1. <Company> - <Title> - <Salary> - <Remote/Boston hybrid>
   Why it fits: <one sentence>
   Apply: <link>
   Watch-outs: <one short note, e.g. travel, meeting load, salary uncertainty>

2. ...

Skipped / watchlist
- <Company> - <Role>: <why skipped>

Reply with feedback labels
- good
- apply
- bad_company
- too_junior
- salary_low
- not_remote
- wrong_sector
- already_seen
- more_like_this: <role/company>
```

Keep the email to 5-10 roles unless there are truly more high-signal matches.

## Feedback loop

Use a lightweight CSV or sheet named `julia_mel_job_feedback.csv` with:

| Column | Purpose |
| --- | --- |
| `date` | Date feedback was given |
| `company` | Company name |
| `title` | Job title |
| `url` | Posting URL |
| `label` | One of the feedback labels |
| `note` | Optional free-text note |

The next scheduled run should:

1. Drop anything labeled `bad_company`, `too_junior`, `salary_low`,
   `not_remote`, `wrong_sector`, or `already_seen`.
2. Promote search terms from `good`, `apply`, and `more_like_this` feedback.
3. Keep a small watchlist for strong roles whose salary or remote status needs
   recruiter confirmation.
4. Include one sentence summarizing how feedback changed the search.

## Validation checklist

- Confirm every top role links to an active employer or trusted job-board page.
- Confirm salary and remote/Boston-hybrid status before applying.
- Confirm any nonprofit near-miss below $120k is intentionally included.
- Keep the CV and any personal contact information out of committed files.

## Rollback

This is a standalone workflow note. To roll back, delete
`julia_mel_daily_job_routine.md`.
