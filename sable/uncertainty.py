"""Uncertainty from five full replies.

For one reply, uncertainty is the mean Shannon entropy of its tokens, in
bits. Each token uses the twenty returned alternatives. Probability that
those alternatives do not cover is one extra outcome, so the entropy is a
lower bound on the full-vocabulary entropy.

The claim's uncertainty is that token mean, averaged over the replies whose
label is the majority label. Agreement is the share of the five replies
that carry the majority label.

The published stability tables stay as they are. Those draws were capped at
24 tokens and did not keep the alternative tokens. This sample generates
the whole reply, at the same temperature 0.7 and top-p 0.9.
"""

from __future__ import annotations

import csv
import math
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = ROOT / "results" / "uncertainty_samples.csv"
SUMMARY_PATH = ROOT / "results" / "uncertainty.csv"
N_SAMPLES = 5
TOP_LOGPROBS = 20
SAMPLE_FIELDS = ["claim_id", "sample", "label", "token_entropy", "n_tokens"]
SUMMARY_FIELDS = ["claim_id", "majority", "agreement", "token_entropy", "majority_draws", "mean_tokens"]


def distribution_entropy(logprobs: list[float]) -> float:
    """Shannon entropy, in bits, of one token. The residual mass is one bin."""
    if not logprobs:
        raise ValueError("empty token distribution")
    probs = [math.exp(value) for value in logprobs]
    total = sum(probs)
    if total <= 0:
        raise ValueError("non-positive token probabilities")
    if total > 1:
        probs = [prob / total for prob in probs]
        total = 1.0
    rest = 1.0 - total
    if rest > 1e-12:
        probs.append(rest)
    return -sum(prob * math.log2(prob) for prob in probs if prob > 0)


def mean_token_entropy(distributions: list[list[float]]) -> float:
    """Mean entropy over every token of one reply."""
    if not distributions:
        raise ValueError("reply has no tokens")
    return sum(distribution_entropy(item) for item in distributions) / len(distributions)


def token_distributions(message) -> list[list[float]]:
    content = (message.response_metadata.get("logprobs") or {}).get("content") or []
    distributions = []
    for token in content:
        alternatives = token.get("top_logprobs") or []
        if not alternatives:
            raise ValueError("top logprobs missing")
        distributions.append([item["logprob"] for item in alternatives])
    if not distributions:
        raise ValueError("reply has no tokens")
    return distributions


def summarize(samples: list[dict]) -> dict:
    """Majority label, agreement, and mean token entropy on the majority replies."""
    if len(samples) != N_SAMPLES:
        raise ValueError(f"expected {N_SAMPLES} replies, got {len(samples)}")
    labels = [sample["label"] for sample in samples]
    counts = Counter(labels)
    top = max(counts.values())
    tied = {label for label, count in counts.items() if count == top}
    majority = next(label for label in labels if label in tied)
    chosen = [sample for sample in samples if sample["label"] == majority]
    return {
        "majority": majority,
        "agreement": top / N_SAMPLES,
        "token_entropy": sum(float(sample["token_entropy"]) for sample in chosen) / len(chosen),
        "majority_draws": len(chosen),
        "mean_tokens": sum(int(sample["n_tokens"]) for sample in chosen) / len(chosen),
    }


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_samples(path: Path | None = None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in _read_csv(SAMPLES_PATH if path is None else path):
        grouped.setdefault(row["claim_id"], []).append(row)
    return grouped


def write_summary(samples: dict[str, list[dict]], path: Path | None = None) -> None:
    path = SUMMARY_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for claim_id in sorted(samples):
        draws = samples[claim_id]
        if len(draws) < N_SAMPLES:
            continue
        summary = summarize(draws[:N_SAMPLES])
        rows.append({"claim_id": claim_id, **summary})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "claim_id": row["claim_id"],
                "majority": row["majority"],
                "agreement": f"{row['agreement']:.1f}",
                "token_entropy": f"{row['token_entropy']:.6f}",
                "majority_draws": row["majority_draws"],
                "mean_tokens": f"{row['mean_tokens']:.1f}",
            })


def _append_sample(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _original_jobs() -> list[dict]:
    import pandas as pd

    from scripts.sample_labels import chunk_parts, prompt_for
    from scripts.sentence_shap import open_store

    claims = pd.read_csv(ROOT / "data" / "CLAIMS.csv").set_index("claim_id")
    predictions = pd.read_csv(ROOT / "results" / "predictions.csv")
    store, k = open_store()
    jobs = []
    for _, row in predictions.iterrows():
        claim_id = row["claim_id"]
        docs = store.similarity_search(claims.loc[claim_id, "statement"], k=k, filter={"source": row["doi"]})
        jobs.append({
            "claim_id": claim_id,
            "prompt": prompt_for(claims.loc[claim_id, "statement"], chunk_parts(docs, None)),
        })
    return jobs


def _draw(llm, prompt: str) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    from scripts.sample_labels import label_and_confidence
    from scripts.sentence_shap import SYSTEM

    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    last_error = None
    for attempt in range(6):
        try:
            message = llm.invoke(messages)
            label, _probability = label_and_confidence(message)
            distributions = token_distributions(message)
            return {
                "label": label,
                "token_entropy": mean_token_entropy(distributions),
                "n_tokens": len(distributions),
            }
        except Exception as error:
            last_error = error
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(last_error)


def run() -> None:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    from sable.tools import ensure_env

    load_dotenv(ROOT / ".env")
    ensure_env()
    jobs = _original_jobs()
    done = load_samples()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        top_p=0.9,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )
    print(f"{len(jobs)} claims", flush=True)
    for index, job in enumerate(jobs, start=1):
        claim_id = job["claim_id"]
        have = done.get(claim_id, [])
        for sample_index in range(len(have), N_SAMPLES):
            drawn = _draw(llm, job["prompt"])
            row = {
                "claim_id": claim_id,
                "sample": sample_index,
                "label": drawn["label"],
                "token_entropy": f"{drawn['token_entropy']:.6f}",
                "n_tokens": drawn["n_tokens"],
            }
            _append_sample(SAMPLES_PATH, row)
            have.append(row)
        done[claim_id] = have
        summary = summarize(have[:N_SAMPLES])
        print(
            f"{index}/{len(jobs)} {claim_id} {summary['majority']} "
            f"agreement={summary['agreement']:.1f} entropy={summary['token_entropy']:.3f}",
            flush=True,
        )
        write_summary(done)
    print(SUMMARY_PATH, flush=True)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
