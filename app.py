import io
import json
import re
import time
from collections import Counter

import numpy as np
import requests
import streamlit as st
from apify_client import ApifyClient

st.set_page_config(page_title="Reel Pulse", page_icon="🥧", layout="centered")

MAX_COMMENTS = 5000
BATCH = 40
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

APIFY_TOKEN = st.secrets["APIFY_TOKEN"]
GROQ_KEY = st.secrets["GROQ_KEY"]
APP_PASSWORD = st.secrets["APP_PASSWORD"]


# ---------- auth gate ----------
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    st.title("🥧 Reel Pulse")
    st.caption("Paste an Instagram reel, get an AI topic breakdown of every comment.")
    pw = st.text_input("Password", type="password")
    if st.button("Enter") or pw:
        if pw == APP_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        elif pw:
            st.error("Wrong password")
    st.stop()


# ---------- helpers ----------
def extract_shortcode(url: str):
    m = re.search(r"instagram\.com/(?:reel|reels|p)/([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def groq(prompt: str, max_tokens: int = 4000) -> str:
    for attempt in range(4):
        resp = requests.post(
            GROQ_BASE,
            headers={"Authorization": f"Bearer {GROQ_KEY}",
                     "Content-Type": "application/json"},
            json={"model": GROQ_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0,
                  "max_tokens": max_tokens},
            timeout=90,
        )
        if resp.status_code in (429, 500, 503):
            time.sleep(8 * (2 ** attempt))
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("Groq API kept failing after retries")


def groq_json(prompt: str, opener: str, closer: str):
    raw = groq(prompt)
    s, e = raw.find(opener), raw.rfind(closer)
    return json.loads(raw[s:e + 1])


def scrape_comments(url: str):
    client = ApifyClient(APIFY_TOKEN)
    run = client.actor("apify/instagram-comment-scraper").call(
        run_input={"directUrls": [url], "resultsLimit": MAX_COMMENTS})
    rows = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        text = (item.get("text") or "").strip()
        rows.append({
            "username": item.get("ownerUsername", ""),
            "text": text,
            "likes": item.get("likesCount", 0) or 0,
        })
    return rows


def derive_taxonomy(comments):
    sample = sorted(comments, key=lambda c: -c["likes"])[:30]
    step = max(1, len(comments) // 90)
    sample += comments[::step][:90]
    numbered = "\n".join(f"- {c['text'][:200]}" for c in sample if c["text"])
    prompt = f"""Below is a sample of comments from one Instagram reel.
Define 5 to 7 broad, mutually exclusive topics that together cover what people are saying.
Generic praise/excitement (including emoji-only positive comments) is usually big enough to be its own topic.
Always include a final catch-all topic named exactly "Other / Tags" for @mentions and off-topic comments.

Sample comments:
{numbered}

Return ONLY a JSON array like:
[{{"name": "Short Topic Name", "desc": "one-line description of what belongs here"}}, ...]"""
    return groq_json(prompt, "[", "]")


def classify(comments, topics, progress):
    topic_list = "\n".join(f"- {t['name']}: {t['desc']}" for t in topics)
    names = {t["name"] for t in topics}
    fallback = "Other / Tags" if "Other / Tags" in names else topics[-1]["name"]
    labels = []
    for i in range(0, len(comments), BATCH):
        batch = comments[i:i + BATCH]
        numbered = "\n".join(f"{j+1}. {c['text'][:300]}" for j, c in enumerate(batch))
        prompt = f"""Classify each Instagram comment into exactly ONE topic (the dominant theme).

Topics:
{topic_list}

Comments:
{numbered}

Return ONLY a JSON array of {len(batch)} strings: the topic name for each comment in order."""
        arr = groq_json(prompt, "[", "]")
        arr = (list(arr) + [fallback] * len(batch))[:len(batch)]
        labels.extend(a if a in names else fallback for a in arr)
        progress.progress(min((i + BATCH) / len(comments), 1.0),
                          text=f"Classifying… {min(i + BATCH, len(comments))}/{len(comments)}")
    return labels


def donut_chart(counts: Counter, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = sum(counts.values())
    items = counts.most_common()
    palette = ["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51",
               "#a06cd5", "#6c91bf", "#d46a9e"]
    colors, ci = [], 0
    for k, _ in items:
        if "other" in k.lower():
            colors.append("#c9c9c9")
        else:
            colors.append(palette[ci % len(palette)]); ci += 1

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    wedges, _ = ax.pie(
        [v for _, v in items], startangle=90, counterclock=False, colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2.5})

    dist_cycle = [1.22, 1.38]
    small_seen = 0
    for w, (lab, v) in zip(wedges, items):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        pct = v / total * 100
        if pct >= 4:
            ax.text(0.79 * x, 0.79 * y, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=15, fontweight="bold", color="white")
        if pct < 11:
            d = dist_cycle[small_seen % 2]; small_seen += 1
        else:
            d = 1.18
        wrapped = lab if len(lab) <= 16 else lab.replace(" / ", "/\n").replace(" & ", " &\n")
        ax.annotate(f"{wrapped}\n{v} comments",
                    xy=(1.02 * x, 1.02 * y), xytext=(d * x, d * y),
                    ha="left" if x >= 0 else "right", va="center",
                    fontsize=12.5, color="#333333", linespacing=1.4,
                    arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 1.0,
                                "shrinkA": 0, "shrinkB": 3})

    ax.text(0, 0.08, f"{total}", ha="center", va="center",
            fontsize=40, fontweight="bold", color="#222222")
    ax.text(0, -0.13, "comments", ha="center", va="center",
            fontsize=15, color="#888888")
    ax.text(0, 1.78, title, ha="center", va="center",
            fontsize=17, fontweight="bold", color="#222222")
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.5, 1.92)
    ax.set_aspect("equal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------- UI ----------
st.title("🥧 Reel Pulse")
st.caption("Paste an Instagram reel URL. Every comment gets scraped and AI-classified "
           f"into broad topics. Capped at {MAX_COMMENTS:,} comments per reel.")

url = st.text_input("Instagram reel URL", placeholder="https://www.instagram.com/reel/…")

if st.button("Analyze", type="primary"):
    code = extract_shortcode(url or "")
    if not code:
        st.error("That doesn't look like an Instagram reel/post URL.")
        st.stop()

    if st.session_state.get("result_code") != code:
        with st.status("Scraping comments… (Instagram rate-limits; this can take 2–5 min)",
                       expanded=False) as status:
            comments = [c for c in scrape_comments(url) if c["text"]]
            status.update(label=f"Scraped {len(comments)} comments", state="complete")
        if not comments:
            st.error("No comments returned. The reel may be private, deleted, or comments are off.")
            st.stop()
        if len(comments) >= MAX_COMMENTS:
            st.warning(f"Hit the {MAX_COMMENTS:,}-comment cap — this is a sample, not the full set.")

        with st.spinner("Deriving topic taxonomy from the comments…"):
            topics = derive_taxonomy(comments)

        progress = st.progress(0.0, text="Classifying…")
        labels = classify(comments, topics, progress)
        progress.empty()

        for c, l in zip(comments, labels):
            c["topic"] = l
        st.session_state.result_code = code
        st.session_state.result_comments = comments
        st.session_state.result_topics = topics

if st.session_state.get("result_comments"):
    comments = st.session_state.result_comments
    code = st.session_state.result_code
    counts = Counter(c["topic"] for c in comments)

    png = donut_chart(counts, f"Comment topics — reel {code}")
    st.image(png)

    st.subheader("Breakdown")
    total = len(comments)
    for t, n in counts.most_common():
        st.write(f"**{t}** — {n} ({n/total*100:.1f}%)")

    csv_buf = io.StringIO()
    csv_buf.write("username,likes,topic,text\n")
    for c in comments:
        text = (c["text"] or "").replace('"', "'").replace("\n", " ")
        csv_buf.write(f'"{c["username"]}",{c["likes"]},"{c["topic"]}","{text}"\n')

    col1, col2 = st.columns(2)
    col1.download_button("⬇ Chart (PNG)", png, f"reel_{code}_topics.png", "image/png")
    col2.download_button("⬇ Comments (CSV)", csv_buf.getvalue(),
                         f"reel_{code}_comments.csv", "text/csv")
