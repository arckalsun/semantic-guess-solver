"""Daily solve entry point: the platform-daily GuessWord challenge.

v0.8.0 (2026-07-15): discovered the visitor-accessible
``/guessV1?date=YYYYMMDD`` endpoint. Previously, getting today's
daily challenge required logging in via WeChat QR (no public API).
With date-mode, any visitor can probe words directly — no
authentication, no share/create round-trip.

This module is the date-mode counterpart to ``round1.py``.

CLI usage::

    python -m sgs.daily_solve \\
        --date 20260715 \\
        --candidates /path/to/cand_words.json \\
        --embeddings /path/to/cand_emb.npy \\
        --batch-size 30 \\
        --rounds 6 \\
        --out /tmp/solve_20260715.ndjson

What it does
------------

1. Build a ``HttpOracle(date=YYYYMMDD)`` (no shareId).
2. Probe an initial seed sweep to lock onto the semantic cluster.
3. Re-rank with ``sgs.rank.rank`` (centroid) until plateau or 100 obs.
4. Switch to ``rank_by_predictor`` (KRR) once:
   - peak < 0.85 after ≥100 probes (centroid plateau)
   - OR peak ≥ 0.85 (already close to answer; KRR refines).
5. Stop on first ``correct=True``.
6. Save the full replay to ``--out``.

The KRR switch is the same one used in case-11 (solved 0.82 plateau
in 9 probes after 633 brute-force rounds). For daily challenges
that cluster cleanly (city/place, food, etc.) KRR is rarely needed —
case-daily-2026-07-15 (Nanning) converged in 150 probes with the
centroid alone.

Probes that fail (rate-limit / network glitch / server-side word
lock) are logged to the NDJSON with ``score=None`` so the ranker
silently skips them — replay can be resumed cleanly by re-running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .rank import load_corpus, rank, rank_by_predictor
from .ratelimit import TokenBucket


HEADERS = {
    "fun-device": "web",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Referer": "https://xiaoce.fun/",
    "accept": "application/json, text/plain, */*",
}


@dataclass
class DailyProbeResult:
    word: str
    score: float | None
    correct: bool = False
    err_code: str | None = None
    err_msg: str | None = None
    raw_score: int | None = None

    def to_ndjson(self) -> dict:
        return {
            "word": self.word,
            "score": self.score,
            "correct": self.correct,
            "doubleScore": self.score,
            "errorCode": self.err_code,
            "errorMessage": self.err_msg,
            "raw_score": self.raw_score,
        }


@dataclass
class DailyOracle:
    """Thin probe wrapper for the date-based daily endpoint.

    Uses urllib stdlib (no requests, no playwright) — same wire as
    ``sgs.wire.http.HttpOracle`` but keyed on ``date`` instead of
    ``shareId``.
    """

    date: str  # yyyyMMdd
    base_url: str = "https://xiaoce.fun"
    timeout_s: float = 8.0

    def probe(self, word: str) -> DailyProbeResult:
        url = (
            f"{self.base_url}/api/v0/quiz/daily/GuessWord/guessV1"
            f"?word={urllib.parse.quote(word, safe='')}"
            f"&date={self.date}"
            f"&skipBusinessErrorToast=true"
        )
        req = urllib.request.Request(url, headers=HEADERS, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                data = json.loads(r.read())
        except Exception as e:
            return DailyProbeResult(
                word=word, score=None,
                err_code="NET", err_msg=str(e),
            )
        body = data.get("data")
        if body is None:
            return DailyProbeResult(
                word=word, score=None,
                err_code=data.get("errorCode"),
                err_msg=data.get("errorMessage"),
            )
        return DailyProbeResult(
            word=word,
            score=body.get("doubleScore"),
            raw_score=body.get("score"),
            correct=body.get("correct", False),
        )


def _seed_sweep() -> list[str]:
    """A broad cluster-coverage seed list. ~30 words across domains.

    The first round is mostly about *not missing* the right cluster.
    We bias toward high-frequency concrete entities (cities, foods,
    objects) because BGE-zh-base places those in well-separated
    regions. Abstract verbs are deliberately under-sampled here —
    they tended to cluster in case-11's 0.82 plateau.
    """
    return [
        # cities / provinces (frequent winners)
        "广州", "上海", "北京", "成都", "武汉", "西安", "天津",
        "重庆", "杭州", "深圳", "南京", "苏州", "青岛", "厦门",
        "南宁", "昆明", "兰州", "拉萨", "银川", "西宁",
        # generic places
        "学校", "医院", "城市", "国家", "山区", "海洋",
        # concrete objects
        "火车", "汽车", "飞机", "电脑", "手机", "电视",
        # foods
        "米饭", "面条", "水果", "饮料", "啤酒", "咖啡",
        # nature
        "山脉", "森林", "彩虹", "海洋",
        # abstract (case-11's plateau neighbors — keep low)
        "通过", "继续", "开始", "结束",
    ]


def _load_observations(log_path: Path, words_in_corpus: set[str]) -> list[tuple[str, float]]:
    """Read past probe records and return (word, score) pairs.

    Words not in the corpus are silently dropped — they were likely
    seed-sweep probes that the ranker can never use.
    """
    obs: list[tuple[str, float]] = []
    if not log_path.exists():
        return obs
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        w = d.get("word")
        s = d.get("score")
        if s is None or w is None:
            continue  # rate-limit or error
        if w not in words_in_corpus:
            continue
        obs.append((w, s))
    return obs


def _record(result: DailyProbeResult, log_path: Path) -> None:
    with log_path.open("a") as f:
        f.write(json.dumps(result.to_ndjson(), ensure_ascii=False) + "\n")


def _print_top(obs: Sequence[tuple[str, float]], n: int = 10) -> None:
    sorted_obs = sorted(obs, key=lambda x: -x[1])[:n]
    peak = sorted_obs[0] if sorted_obs else ("", 0.0)
    print(f"  top-{n}: {sorted_obs} | peak={peak[1]:.4f}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m sgs.daily_solve",
        description="Daily GuessWord solver (visitor, date-based).",
    )
    p.add_argument("--date", required=True, help="yyyyMMdd (e.g., 20260715)")
    p.add_argument(
        "--candidates", type=Path, required=True,
        help="JSON list[str] of candidate words.",
    )
    p.add_argument(
        "--embeddings", type=Path, required=True,
        help="(N, D) float32 .npy aligned with --candidates.",
    )
    p.add_argument(
        "--batch-size", type=int, default=30,
        help="Probe batch size (default 30, matches UI batch).",
    )
    p.add_argument(
        "--rounds", type=int, default=10,
        help="Max probe rounds (default 10, ~300 probes).",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("/tmp/daily_solve.ndjson"),
        help="NDJSON replay path (default /tmp/daily_solve.ndjson).",
    )
    p.add_argument(
        "--seed", type=str, default=None,
        help="Custom seed list (comma-separated words). "
             "Default uses built-in _seed_sweep.",
    )
    p.add_argument(
        "--rate", type=float, default=0.8,
        help="Token bucket rate (probes/sec, default 0.8).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Daily solve: {args.date} ===")
    words, emb = load_corpus(args.candidates, args.embeddings)
    words_list = list(words)
    words_set = set(words)
    print(f"corpus: {len(words_list)} words, emb shape {emb.shape}")

    oracle = DailyOracle(date=args.date)
    bucket = TokenBucket(rate=args.rate, burst=2)

    obs = _load_observations(args.out, words_set)
    used = set(w for w, _ in obs)
    print(f"resumed with {len(obs)} obs (file: {args.out})")

    # ---- Phase 0: seed sweep (only if no obs yet) ----
    if not obs:
        seed = args.seed.split(",") if args.seed else _seed_sweep()
        for w in seed:
            if w in used or w not in words_set:
                # skip; probe missing-corpus words anyway to record + learn
                pass
            bucket.consume()
            result = oracle.probe(w)
            _record(result, args.out)
            score_str = f"{result.score:.4f}" if result.score is not None else f"err={result.err_code}"
            correct_str = "✓" if result.correct else ""
            print(f"  [seed] {w:>6} → {score_str} {correct_str}")
            if result.correct:
                print(f"\n!!! CORRECT on seed: {w} !!!")
                return 0
            if result.score is not None:
                obs.append((w, result.score))
            used.add(w)
            time.sleep(0.3)
        # reload after seed
        obs = _load_observations(args.out, words_set)
        _print_top(obs)

    # ---- Main loop ----
    for round_n in range(1, args.rounds + 1):
        n_obs = len(obs)
        peak = max((s for _, s in obs), default=0.0)
        use_krr = (n_obs >= 100) or (peak >= 0.85)
        mode = "KRR" if use_krr else "centroid"
        print(f"\n--- Round {round_n}: {n_obs} obs, peak={peak:.4f}, mode={mode} ---")

        if not obs:
            print("no observations, stopping")
            break

        ranked = (
            rank_by_predictor(obs, words_list, emb, top_k=args.batch_size * 2)
            if use_krr
            else rank(obs, words_list, emb, top_k=args.batch_size * 2)
        )

        picked: list[tuple[str, float]] = []
        for w, sim in ranked:
            if w not in used:
                picked.append((w, sim))
            if len(picked) >= args.batch_size:
                break

        if not picked:
            print("corpus exhausted — no unprobed candidates remain")
            break

        for w, sim in picked:
            bucket.consume()
            result = oracle.probe(w)
            _record(result, args.out)
            score = result.score
            correct = result.correct
            err = result.err_code or ""
            extra = "✓ CORRECT" if correct else (f"err={err}" if err else "")
            score_str = f"{score:.4f}" if score is not None else "n/a"
            print(f"  [{n_obs + len(picked) - len(used) + 1:>3}] {w:>6} sim={sim:.4f} → {score_str} {extra}")
            if correct:
                print(f"\n!!! CORRECT: {w} !!!")
                return 0
            if score is not None:
                obs.append((w, score))
            used.add(w)
            time.sleep(0.3)

        _print_top(obs)

        if any(s >= 0.99 for _, s in obs):
            print("KRR achieves near-perfect confidence — final answer imminent")
            # fall through to the next round; KRR will surface it

    # ---- Loop ended without CORRECT ----
    obs_sorted = sorted(obs, key=lambda x: -x[1])
    peak_word, peak_score = obs_sorted[0]
    print(f"\n=== Finished without finding answer ===")
    print(f"Best score: {peak_word} = {peak_score:.4f}")
    print(f"Re-run with --rounds={args.rounds * 2} to continue, or")
    print(f"inspect the top words manually. Replay at: {args.out}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
