import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import re
import json
import spacy
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher

st.set_page_config(page_title="Lead Extractor", layout="wide")

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, sbert

nlp, sbert = load_models()

MAX_PAGES = 15

ROLE_PRIORITY = {
    "CEO": 12, "OWNER": 12, "FOUNDER": 11, "CO-FOUNDER": 11,
    "MANAGING DIRECTOR": 10, "PRESIDENT": 9,
    "CTO": 7, "CFO": 7, "COO": 7,
    "DIRECTOR": 6, "VP": 5,
    "HEAD": 4, "MANAGER": 3,
}

ROLE_KEYWORDS = [
    ("MANAGING DIRECTOR", ["managing director"]),
    ("CO-FOUNDER", ["co-founder", "cofounder", "co founder"]),
    ("CEO", ["chief executive officer", "ceo"]),
    ("OWNER", ["owner"]),
    ("FOUNDER", ["founder"]),
    ("PRESIDENT", ["president"]),
    ("CTO", ["chief technology officer", "cto"]),
    ("CFO", ["chief financial officer", "cfo"]),
    ("COO", ["chief operating officer", "coo"]),
    ("DIRECTOR", ["director"]),
    ("VP", ["vice president", "vp"]),
    ("HEAD", ["head of"]),
    ("MANAGER", ["manager"]),
]

ROLE_EXTRA_KEYWORDS = [
    "consultant", "officer", "recruiter", "coordinator",
    "specialist", "executive", "associate", "analyst",
    "partner", "principal", "lead", "advisor"
]

NAME_BLACKLIST = {
    "register", "login", "home", "menu", "search", "read", "more",
    "view", "blog", "news", "support", "contact", "about",
    "services", "courses", "training", "learn", "download",
    "privacy", "policy", "terms", "team", "welcome"
}

SOURCE_WEIGHT = {
    "json_ld": 4,
    "html_sibling": 3,
    "html_line": 2,
    "spacy": 1,
}

def normalize_url(url):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")

def same_domain(base, link):
    return urlparse(base).netloc in urlparse(link).netloc

def clean_name(raw):
    name = re.sub(r"\S+@\S+", "", raw)
    name = re.sub(r"[^a-zA-Z\s'\-]", "", name)
    name = " ".join(name.split())
    parts = name.split()

    if not (2 <= len(parts) <= 4):
        return None

    if any(p.lower() in NAME_BLACKLIST for p in parts):
        return None

    if any(len(p) < 2 for p in parts):
        return None

    if any(p.isupper() and len(p) > 2 for p in parts):
        return None

    if not all(p[0].isupper() for p in parts if p[0].isalpha()):
        return None

    return name

def detect_all_roles(text):
    t = text.lower()
    found = []

    for role, keys in ROLE_KEYWORDS:
        if any(k in t for k in keys):
            found.append(role)

    found = list(set(found))
    found.sort(key=lambda x: ROLE_PRIORITY.get(x, 0), reverse=True)
    return found

def best_role(text):
    roles = detect_all_roles(text)
    return roles[0] if roles else None

def is_role_text(text):
    t = text.lower()
    all_keys = [k for _, keys in ROLE_KEYWORDS for k in keys]
    return any(k in t for k in all_keys + ROLE_EXTRA_KEYWORDS)

def normalize_role(raw):
    if not raw:
        return None
    roles = detect_all_roles(raw)
    if roles:
        return roles[0]
    return raw.strip().title()

def email_match_score(name, email):
    parts = [p.lower() for p in name.split()]
    user = re.sub(r"[^a-z]", "", email.split("@")[0].lower())

    score = 0.0

    if len(parts) >= 2 and parts[0] in user and parts[-1] in user:
        return 5.0

    if len(parts) >= 2:
        initials_last = parts[0][0] + parts[-1]
        if initials_last in user:
            return 3.5

    matching = [p for p in parts if p in user]
    score += len(matching) * 2

    ratio = SequenceMatcher(None, "".join(parts), user).ratio()
    score += ratio * 1.5

    if score < 1:
        nv = sbert.encode(name, convert_to_tensor=True)
        uv = sbert.encode(user, convert_to_tensor=True)
        score += util.cos_sim(nv, uv).item()

    return score

def match_best_email(name, emails):
    if not emails:
        return None, 0

    scored = [(e, email_match_score(name, e)) for e in emails]
    best = max(scored, key=lambda x: x[1])

    if best[1] > 0.8:
        return best
    return None, 0

def candidate_score(c):
    role_pts = ROLE_PRIORITY.get(c["role"], 1) if c["role"] else 0
    source_pts = SOURCE_WEIGHT.get(c["source"], 1)
    email_pts = c["email_score"]
    name_bonus = 1 if len(c["name"].split()) == 3 else 0
    return role_pts + source_pts + email_pts + name_bonus

def extract_from_json_ld(soup):
    people = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            items = data if isinstance(data, list) else [data]

            for item in items:
                if not isinstance(item, dict):
                    continue

                for node in item.get("@graph", [item]):
                    if not isinstance(node, dict):
                        continue

                    if node.get("@type") == "Person":
                        name = clean_name(node.get("name", ""))

                        if not name:
                            continue

                        role = normalize_role(node.get("jobTitle", ""))
                        email = node.get("email", "").replace("mailto:", "").strip() or None

                        people.append({
                            "name": name,
                            "role": role,
                            "email": email,
                            "email_score": 0,
                            "source": "json_ld"
                        })
        except:
            pass

    return people

def extract_from_html_structure(soup):
    people = []
    seen = set()

    for tag in soup.find_all(["h2", "h3", "h4", "h5"]):
        raw = tag.get_text(strip=True)
        name = clean_name(raw)

        if not name or name.lower() in seen:
            continue

        role_text = None

        nxt = tag.find_next_sibling()
        if nxt:
            t = nxt.get_text(strip=True)
            if is_role_text(t):
                role_text = t

        if role_text:
            people.append({
                "name": name,
                "role": normalize_role(role_text),
                "email": None,
                "email_score": 0,
                "source": "html_sibling"
            })
            seen.add(name.lower())

    return people

def extract_from_line_scan(soup, seen_names):
    people = []
    seen = set(seen_names)

    lines = [l.strip() for l in soup.get_text("\n", strip=True).split("\n") if l.strip()]

    for i, line in enumerate(lines):
        name = clean_name(line)

        if not name or name.lower() in seen:
            continue

        for j in range(1, 4):
            if i + j >= len(lines):
                break

            next_line = lines[i + j]

            if is_role_text(next_line):
                people.append({
                    "name": name,
                    "role": normalize_role(next_line),
                    "email": None,
                    "email_score": 0,
                    "source": "html_line"
                })
                seen.add(name.lower())
                break

    return people

def extract_from_spacy(text, seen_names):
    people = []
    seen = set(seen_names)

    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue

        name = clean_name(ent.text)

        if not name or name.lower() in seen:
            continue

        after = text[ent.end_char: ent.end_char + 150]
        role = best_role(after)

        if role:
            people.append({
                "name": name,
                "role": role,
                "email": None,
                "email_score": 0,
                "source": "spacy"
            })
            seen.add(name.lower())

    return people

def get_html(page, url):
    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)
        return page.content()
    except:
        return ""

def extract_emails(text, soup=None):
    found = set(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))

    if soup:
        for a in soup.find_all("a", href=True):
            if a["href"].startswith("mailto:"):
                found.add(a["href"].replace("mailto:", "").split("?")[0])

    return list(found)

def extract_phone(text):
    patterns = [
        r"\+\d{1,3}[\s\-]?\d{3,5}[\s\-]?\d{3,5}[\s\-]?\d{2,5}",
        r"\b0\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}\b",
        r"\b\d{10}\b"
    ]

    found = []

    for pat in patterns:
        found.extend(re.findall(pat, text))

    return found[0] if found else None

def extract_linkedin(soup):
    for a in soup.find_all("a", href=True):
        if "linkedin.com/company" in a["href"]:
            return a["href"].split("?")[0]
    return None

def extract_company(soup):
    og = soup.find("meta", property="og:site_name")
    if og and og.get("content"):
        return og["content"].strip()

    if soup.title:
        return soup.title.text.strip()

    return None

def best_general_email(emails):
    priority = ["info", "contact", "hello", "support", "sales"]

    for p in priority:
        for e in emails:
            if p in e.lower():
                return e

    return emails[0] if emails else None

def extract_lead(url):
    url = normalize_url(url)
    browser = p.chromium.launch(
    headless=True,
    args=[
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--single-process"
    ]
)
    visited = set()
    queued = {url}
    queue = [url]

    company = None
    linkedin = None
    phone = None

    all_emails = set()
    all_candidates = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
    headless=True,
    args=[
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--single-process"
    ]
)
        ctx = browser.new_context()
        page = ctx.new_page()

        while queue and len(visited) < MAX_PAGES:
            link = queue.pop(0)

            if link in visited:
                continue

            visited.add(link)

            html = get_html(page, link)

            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)

            all_emails.update(extract_emails(text, soup))

            if not company:
                company = extract_company(soup)

            if not linkedin:
                linkedin = extract_linkedin(soup)

            if not phone:
                phone = extract_phone(text)

            page_candidates = []
            page_candidates += extract_from_json_ld(soup)
            page_candidates += extract_from_html_structure(soup)

            seen = {c["name"].lower() for c in page_candidates}
            page_candidates += extract_from_line_scan(soup, seen)

            seen = {c["name"].lower() for c in page_candidates}
            page_candidates += extract_from_spacy(text, seen)

            all_candidates += page_candidates

            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])

                if same_domain(url, href) and href not in queued:
                    if any(k in href.lower() for k in ["about", "team", "contact", "leadership", "staff"]):
                        queue.append(href)
                        queued.add(href)

        ctx.close()
        browser.close()

    emails = list(all_emails)

    for c in all_candidates:
        if not c["email"]:
            e, sc = match_best_email(c["name"], emails)
            c["email"] = e
            c["email_score"] = sc
        else:
            c["email_score"] = email_match_score(c["name"], c["email"])

    merged = {}

    for c in all_candidates:
        key = c["name"].lower()

        if key not in merged:
            merged[key] = c
        else:
            if ROLE_PRIORITY.get(c["role"], 0) > ROLE_PRIORITY.get(merged[key]["role"], 0):
                merged[key]["role"] = c["role"]

            if c["email_score"] > merged[key]["email_score"]:
                merged[key]["email"] = c["email"]
                merged[key]["email_score"] = c["email_score"]

    candidates = list(merged.values())
    candidates.sort(key=candidate_score, reverse=True)

    dm = candidates[0] if candidates else None

    return {
        "Company": company,
        "Phone": phone,
        "Email": best_general_email(emails),
        "LinkedIn": linkedin,
        "Decision Maker": dm
    }

st.title("Lead Extraction System")

url = st.text_input("Enter Website URL")

if st.button("Extract Lead"):
    if url:
        with st.spinner("Extracting..."):
            result = extract_lead(url)

        st.success("Completed")

        st.subheader("Company Details")
        st.write("Company:", result["Company"])
        st.write("Phone:", result["Phone"])
        st.write("Email:", result["Email"])
        st.write("LinkedIn:", result["LinkedIn"])

        st.subheader("Decision Maker")

        dm = result["Decision Maker"]

        if dm:
            st.write("Name:", dm["name"])
            st.write("Role:", dm["role"])
            st.write("Email:", dm["email"])
        else:
            st.write("Not Found")
