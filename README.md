# Geopolitical News Intelligence System — MVP v1

A fully automated, zero-cost, local-first pipeline that ingests global news,
clusters it into real-world events, and delivers structured daily email briefings.

---

## System Requirements

| Requirement | Value |
|---|---|
| OS | Windows (also works on Linux/macOS) |
| Python | 3.12.2 |
| Ollama | Installed and running locally |
| LLM model | `qwen:4b` via Ollama |

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd geopolitical_intel
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Install Ollama (Windows)

Download from: https://ollama.com/download/windows

Then pull the Qwen model:

```bash
ollama pull qwen:4b
```

Verify Ollama is running:

```bash
ollama list   # should show qwen:4b
```

### 4. Configure credentials

All configuration lives in `config/config.py`.

The following are pre-configured from the PRD:

```python
EMAIL_SENDER   = "batmanintj@gmail.com"
EMAIL_PASSWORD = "lmgp levd payc vvpn"   # Gmail App Password
EMAIL_RECEIVER = "sshafs2004@gmail.com"
```

> **Note:** The `EMAIL_PASSWORD` is a Gmail App Password (16-char), not the
> account password. Ensure "Less secure app access" or App Passwords are
> enabled in the Gmail account settings.

---

## Running the Pipeline

### Manual run

```bash
cd geopolitical_intel
python main.py
```

### Debug mode (shows intermediate outputs, clusters, selected sentences)

```bash
python main.py --debug
```

---

## Project Structure

```
geopolitical_intel/
│
├── config/
│   └── config.py              # All hyperparameters — edit here, not in code
│
├── ingestion/
│   └── rss_fetcher.py         # RSS fetch with retry + cache fallback
│
├── preprocessing/
│   └── preprocessor.py        # HTML clean, boilerplate, normalize, segment, filter
│
├── deduplication/
│   └── deduplicator.py        # URL dedup → hash dedup → semantic dedup
│
├── classification/
│   └── classifier.py          # DistilBERT zero-shot geopolitical classifier
│
├── embedding/
│   └── embedder.py            # Sliding window chunking + MiniLM embedding
│
├── clustering/
│   └── clusterer.py           # HDBSCAN + fallback chain
│
├── ranking/
│   └── ranker.py              # Score clusters, select Top 6
│
├── summarization/
│   ├── mmr.py                 # MMR article selection
│   ├── textrank.py            # TextRank extractive compression
│   ├── sentence_dedup.py      # Cosine sentence deduplication
│   ├── fact_selector.py       # Token budget enforcement (7-12 facts)
│   ├── abstractive.py         # Qwen paragraph generation
│   ├── title_generator.py     # Qwen title generation
│   └── summarizer.py          # Orchestrates all summarization steps
│
├── email/
│   ├── formatter.py           # Assembles exact PRD email format
│   └── sender.py              # Gmail SMTP + developer error alerts
│
├── utils/
│   ├── logger.py              # Per-stage logging → logs/YYYY-MM-DD.log
│   └── helpers.py             # Shared: timer, normalize, cosine_sim, chunking
│
├── storage/
│   └── embedding_store.py     # Persistent local cache (TTL: 3 days)
│
├── logs/                      # Auto-created; one log file per run date
├── main.py                    # Pipeline orchestrator
└── requirements.txt
```

---

## Pipeline Flow

```
RSS Feeds
    │
    ▼
[1] RSS Ingestion          feedparser, retry×3, RSS cache fallback
    │
    ▼
[2] Preprocessing          trafilatura → newspaper3k → html strip
                           spaCy sentence segmentation, word_count filter
    │
    ▼
[3] Deduplication          URL → SHA-256 hash → cosine similarity (≥0.85)
                           Cross-day state persisted to storage/
    │
    ▼
[4] Classification         distilbert-base-uncased zero-shot
                           threshold = 0.5
    │
    ▼
[5] Chunking + Embedding   250-word sliding window, 50-word overlap
                           all-MiniLM-L6-v2 per chunk
    │
    ▼
[6] Clustering             HDBSCAN (min_cluster_size=3)
                           Fallback: size-2 pairs → high-confidence singles
    │
    ▼
[7] Ranking                score = 0.5×size + 0.3×recency + 0.2×diversity
                           Top 6 selected
    │
    ▼
[8] Summarization
      ├── MMR article selection   (k=5, λ=0.7)
      ├── TextRank extraction     (top 5 sentences/article)
      ├── Sentence dedup          (cosine ≥ 0.85 → remove)
      ├── Fact selection          (7–12 sentences, frequency + centrality)
      ├── Qwen paragraph          (≤120 words, grounded from bullets only)
      └── Qwen title              (≤12 words, trusted sources + bullets)
    │
    ▼
[9] Email Formatting       Exact PRD structure, plain text
    │
    ▼
[10] Email Delivery        Gmail SMTP, TLS port 587
```

---

## Email Output Format

```
EVENT 1: <Generated Title>
────────────────────────────────────────────────────────────────────────

<Abstractive Summary Paragraph (≤120 words)>

Extractive Facts:
  - Fact 1
  - Fact 2
  ...
  - Fact N (7–12 bullets)

Sources:
  - URL 1
  - URL 2
  - URL 3
```

---

## Configuration Reference

All values in `config/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `DEBUG` | `False` | Print intermediate pipeline outputs |
| `MIN_WORD_COUNT` | `100` | Minimum article length |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.85` | Semantic dedup cosine threshold |
| `CLASSIFICATION_THRESHOLD` | `0.5` | Min geo confidence to keep article |
| `CHUNK_SIZE` | `250` | Words per embedding chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MIN_CLUSTER_SIZE` | `3` | HDBSCAN primary min cluster size |
| `TOP_N_EVENTS` | `6` | Events per email |
| `MMR_K` | `5` | Articles selected per cluster (MMR) |
| `MMR_LAMBDA` | `0.7` | MMR relevance weight |
| `TEXTRANK_TOP_SENTENCES` | `5` | Sentences extracted per article |
| `SENTENCE_SIM_THRESHOLD` | `0.85` | Cosine threshold for sentence dedup |
| `MIN_SUMMARY_SENTENCES` | `7` | Minimum facts per event |
| `MAX_SUMMARY_SENTENCES` | `12` | Maximum facts per event |
| `QWEN_PARAGRAPH_MAX_WORDS` | `120` | Max paragraph length |
| `TITLE_MAX_WORDS` | `12` | Max title length |
| `STORAGE_TTL_DAYS` | `3` | Days before purging local storage |

---

## Logging

Each run writes to `logs/YYYY-MM-DD.log`:

- Articles fetched per feed
- Duplicates removed (each pass)
- Articles classified
- Clusters formed and selected
- Summarization success/failure per event
- Runtime per stage
- Full stack traces on errors

---

## Failure Handling

| Case | Behaviour |
|---|---|
| RSS feed down | Retry ×3, then use last cached fetch |
| No articles found | Send "Insufficient data" notice email |
| Clustering failure | Top-N articles summarized individually |
| LLM (Qwen) failure | Send bullet-only summaries (no paragraph) |
| Any pipeline error | Send stack trace to developer email |

---

## Scheduling (GitHub Actions)

The workflow runs at **7 AM and 7 PM local time (PKT = UTC+5)**,
mapping to 02:00 and 14:00 UTC.

To adjust for your timezone, edit `.github/workflows/briefing.yml`:

```yaml
- cron: "0 2,14 * * *"   # 02:00 and 14:00 UTC = 07:00 and 19:00 PKT
```

Manual trigger is also available from the GitHub Actions UI.
