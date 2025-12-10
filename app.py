# app.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

import config
from cache_disk import DiskCache
from data_prices import fetch_prices
import os
from dotenv import load_dotenv

from data_gnews import fetch_latest_headlines
from data_gdelt import (
    build_keyword_query as build_gdelt_query,
    fetch_gdelt_metrics,
    fetch_latest_headlines as fetch_gdelt_headlines,
)
from scoring import (
    compute_price_features,
    fetch_gnews_for_universe,
    fetch_x_for_tickers,
    rank_stage1,
    rank_final,
)

load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "").strip()
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()



st.set_page_config(page_title="IDX Sentiment Screener", layout="wide")

cache = DiskCache(Path(__file__).resolve().parent / ".cache")

st.title("IDX Sentiment Screener (GDELT + yfinance)")
st.caption("Runs on demand and caches results (disk + Streamlit cache).")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")
    advanced_mode = st.checkbox("Advanced mode", value=False)

    st.subheader("Cache")
    c1, c2 = st.columns(2)
    if c1.button("Clear disk cache"):
        n = cache.clear()
        st.success(f"Cleared {n} cached files.")
    if c2.button("Clear Streamlit cache"):
        st.cache_data.clear()
        st.success("Cleared Streamlit cache.")

    st.subheader("Data sources")
    price_mode_ui = st.selectbox(
        "Price source",
        ["Yahoo (fast)", "Yahoo with Stooq fallback", "Stooq only"],
        index=1,
    )
    mode_map = {
        "Yahoo (fast)": "yahoo",
        "Yahoo with Stooq fallback": "yahoo_fallback_stooq",
        "Stooq only": "stooq",
    }

    st.subheader("Ranking")
    top_n = st.number_input("Top N", min_value=3, max_value=30, value=config.TOP_N_DEFAULT, step=1)
    min_vol = st.number_input(
        "Min avg daily volume (shares)",
        min_value=0,
        value=config.MIN_AVG_DAILY_VOLUME_DEFAULT,
        step=500_000,
    )

    tone_only = st.checkbox("Tone only (faster, fewer API calls)", value=False)
    show_headlines = st.checkbox("Show headlines (slower)", value=True)

    if advanced_mode:
        st.subheader("Filters")
        prefilter_n = st.number_input(
            "Prefilter N (liquid first)",
            min_value=10,
            max_value=300,
            value=config.PREFILTER_N_DEFAULT,
            step=5,
        )

        gnews_cap = st.number_input(
            "Max tickers for GNews (rate limit)",
            min_value=5,
            max_value=100,
            value=config.GNEWS_MAX_TICKERS_DEFAULT,
            step=1,
            help="Only the most liquid tickers will hit GNews to avoid 429 throttling; others fall back to GDELT.",
        )

        st.subheader("Lookback")
        lookback_news = st.number_input(
            "News lookback (days)",
            min_value=1,
            max_value=30,
            value=config.LOOKBACK_DAYS_NEWS_DEFAULT,
            step=1,
        )
        lookback_price = st.number_input(
            "Price lookback (days)",
            min_value=10,
            max_value=365,
            value=config.LOOKBACK_DAYS_PRICE_DEFAULT,
            step=5,
        )

        st.subheader("Speed")
        gdelt_workers = st.slider("GDELT workers", 2, 20, config.GDELT_WORKERS_DEFAULT, 1)

        st.subheader("Weights")
        w_tone = st.slider("Tone weight", 0.0, 1.0, config.WEIGHT_TONE_DEFAULT, 0.05)
        w_vol = st.slider("News volume weight", 0.0, 1.0, config.WEIGHT_NEWS_VOL_DEFAULT, 0.05)
        w_mom = st.slider("Momentum weight", 0.0, 1.0, config.WEIGHT_MOMENTUM_DEFAULT, 0.05)
    else:
        prefilter_n = config.PREFILTER_N_DEFAULT
        gnews_cap = config.GNEWS_MAX_TICKERS_DEFAULT
        lookback_news = config.LOOKBACK_DAYS_NEWS_DEFAULT
        lookback_price = config.LOOKBACK_DAYS_PRICE_DEFAULT
        gdelt_workers = config.GDELT_WORKERS_DEFAULT
        w_tone = config.WEIGHT_TONE_DEFAULT
        w_vol = config.WEIGHT_NEWS_VOL_DEFAULT
        w_mom = config.WEIGHT_MOMENTUM_DEFAULT


# ----------------------------
# Main inputs
# ----------------------------
universe_text = st.text_area(
    "Universe (IDX tickers, one per line, no .JK)",
    value="\n".join(config.WATCHLIST),
    height=180,
)

tickers = [t.strip().upper() for t in universe_text.splitlines() if t.strip()]
tickers_jk = [f"{t}.JK" for t in tickers]

run = st.button("Run screening", type="primary")

# ----------------------------
# Run pipeline with progress bar
# ----------------------------
if run:
    # Progress UI
    pb_slot = st.empty()
    pbar = pb_slot.progress(0, text="Starting...")

    def set_progress(pct: int, text: str) -> None:
        pct = int(max(0, min(100, pct)))
        pbar.progress(pct, text=text)

    # 1) Prices
    set_progress(5, "Fetching prices...")
    price_df, src_map = fetch_prices(
        tickers_jk=tickers_jk,
        lookback_days_price=int(lookback_price),
        cache=cache,
        ttl_seconds=config.CACHE_TTL_SECONDS,
        mode=mode_map[price_mode_ui],
        yf_threads=True,          # speed again
        stooq_overrides={},       # you can add per-ticker overrides later
    )

    src_counts = pd.Series(list(src_map.values())).value_counts()
    y_ct = int(src_counts.get("yahoo", 0))
    s_ct = int(src_counts.get("stooq", 0))
    n_ct = int(src_counts.get("none", 0))
    st.info(f"Prices fetched: Yahoo={y_ct}, Stooq={s_ct}, Missing={n_ct}")

    # 2) Price features (or sentiment-only fallback)
    sentiment_only_mode = bool(price_df is None or price_df.empty)
    if sentiment_only_mode:
        st.warning("No price data available. Using sentiment-only ranking (momentum weight forced to 0).")
        w_mom = 0.0
        set_progress(20, "Building minimal features (sentiment-only)...")
        feat_df = pd.DataFrame(
            {
                "ticker": [t.replace(".JK", "") for t in tickers_jk],
                "ticker_jk": tickers_jk,
                "last_close": np.nan,
                "ret_5d": 0.0,
                "avg_vol_20d": np.nan,
            }
        )
    else:
        set_progress(20, "Computing price features...")
        feat_df = compute_price_features(price_df)

    # 3) Liquidity filter, then prefilter
    set_progress(30, "Applying liquidity filter...")
    f = feat_df.copy()
    f["avg_vol_20d"] = pd.to_numeric(f["avg_vol_20d"], errors="coerce")
    f = f[(f["avg_vol_20d"].isna()) | (f["avg_vol_20d"] >= float(min_vol))]
    if f.empty:
        f = feat_df.copy()

    f = (
        f.sort_values(["avg_vol_20d", "ret_5d"], ascending=[False, False])
        .head(int(prefilter_n))
        .reset_index(drop=True)
    )

    # 4) Stage-1: GNews sentiment for most-liquid prefiltered tickers (cap to avoid 429s)
    gnews_targets = f.head(int(gnews_cap)).copy()
    skipped_for_gnews = f.iloc[len(gnews_targets) :]["ticker"].tolist()
    total_news = len(gnews_targets)

    gnews_errors: list[str] = []

    def gnews_progress(done: int, total: int, ticker: str | None = None) -> None:
        total = max(1, int(total))
        done = int(done)
        pct = 30 + int(40 * (done / total))  # 30..70
        name = f" ({ticker})" if ticker else ""
        set_progress(pct, f"GNews sentiment {done}/{total}{name}...")

    if not GNEWS_API_KEY:
        st.warning("GNEWS_API_KEY not set. GNews stage will return empty -> ranking will be weak.")
        gnews_df = pd.DataFrame(columns=["ticker", "kw_used", "gnews_sent", "gnews_buzz"])
    else:
        if skipped_for_gnews:
            gnews_errors.append(
                f"Skipping GNews for {len(skipped_for_gnews)} tickers beyond cap {gnews_cap}; they will use GDELT fallback if needed."
            )

        set_progress(30, f"Fetching GNews for {total_news} tickers...")
        gnews_df = fetch_gnews_for_universe(
            tickers=gnews_targets["ticker"].tolist(),
            aliases=config.ALIASES,
            gnews_api_key=GNEWS_API_KEY,
            lookback_days_news=int(lookback_news),
            max_articles_per_ticker=10,
            cache=cache,
            ttl_seconds=config.CACHE_TTL_SECONDS,
            max_workers=int(gdelt_workers),
            progress_cb=gnews_progress,
            error_log=gnews_errors,
        )

        if skipped_for_gnews:
            pad = pd.DataFrame(
                {
                    "ticker": skipped_for_gnews,
                    "kw_used": [""] * len(skipped_for_gnews),
                    "gnews_sent": [np.nan] * len(skipped_for_gnews),
                    "gnews_buzz": [0] * len(skipped_for_gnews),
                }
            )
            gnews_df = pd.concat([gnews_df, pad], ignore_index=True)

        # Fallback to GDELT for tickers where GNews returned nothing (HTTP 429, empty, etc.)
        missing_tickers = [
            str(row["ticker"])
            for _, row in gnews_df.fillna({"gnews_buzz": 0}).iterrows()
            if (row.get("gnews_buzz", 0) == 0 and pd.isna(row.get("gnews_sent")))
        ]

        if missing_tickers:
            gnews_errors.append(
                f"Falling back to GDELT for {len(missing_tickers)} tickers with no GNews coverage."
            )

            def gdelt_progress(done: int, total: int, ticker: str | None = None) -> None:
                total = max(1, int(total))
                done = int(done)
                pct = 30 + int(15 * (done / total))  # 30..45
                name = f" ({ticker})" if ticker else ""
                set_progress(pct, f"GDELT fallback {done}/{total}{name}...")

            gdelt_rows = []
            total_missing = max(1, len(missing_tickers))
            for i, t in enumerate(missing_tickers, start=1):
                gdelt_progress(i, total_missing, t)
                q = build_gdelt_query(t, config.ALIASES)
                metrics = fetch_gdelt_metrics(
                    query=q,
                    lookback_days_news=int(lookback_news),
                    include_vol=True,
                    cache=cache,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )
                gdelt_rows.append(
                    {
                        "ticker": t,
                        "kw_used": q,
                        "gnews_sent": metrics.get("tone_latest", np.nan),
                        "gnews_buzz": metrics.get("vol_latest", 0),
                    }
                )

            if gdelt_rows:
                gdf = gnews_df.set_index("ticker")
                for r in gdelt_rows:
                    t = r["ticker"]
                    if t not in gdf.index:
                        gdf.loc[t] = {"kw_used": "", "gnews_sent": np.nan, "gnews_buzz": 0}
                    if pd.isna(gdf.loc[t].get("gnews_sent")) and gdf.loc[t].get("gnews_buzz", 0) == 0:
                        gdf.loc[t, ["kw_used", "gnews_sent", "gnews_buzz"]] = [
                            r["kw_used"],
                            r["gnews_sent"],
                            r["gnews_buzz"],
                        ]
                gnews_df = gdf.reset_index()

    merged = f.merge(gnews_df, on="ticker", how="left")

    # If user checked "Tone only", interpret it as: GNews only (skip X stage)
    # (You can rename the checkbox label later if you want.)
    set_progress(72, "Ranking stage-1 (GNews + momentum)...")
    stage1 = rank_stage1(
        merged,
        w_sent=float(w_tone),  # reuse slider: "Tone weight" -> GNews sentiment
        w_buzz=float(w_vol),   # reuse slider: "News volume weight" -> GNews buzz
        w_mom=float(w_mom),
    )

    # 5) Stage-2: X sentiment for only Top 10 from stage-1
    top_x_n = 10
    top_for_x = stage1.head(min(top_x_n, len(stage1)))["ticker"].tolist()

    def x_progress(done: int, total: int, ticker: str | None = None) -> None:
        total = max(1, int(total))
        done = int(done)
        pct = 70 + int(18 * (done / total))  # 70..88
        name = f" ({ticker})" if ticker else ""
        set_progress(pct, f"X sentiment {done}/{total}{name}...")

    x_df = pd.DataFrame(columns=["ticker", "x_query", "x_sent", "x_buzz"])

    if tone_only:
        st.info("Skipping X stage (GNews-only mode).")
    elif not X_BEARER_TOKEN:
        st.info("X stage skipped (X_BEARER_TOKEN not set).")
    elif top_for_x:
        set_progress(75, f"Fetching X for Top {len(top_for_x)}...")
        x_df = fetch_x_for_tickers(
            tickers=top_for_x,
            aliases=config.ALIASES,
            x_bearer_token=X_BEARER_TOKEN,
            lookback_hours=72,
            max_posts_per_ticker=50,
            cache=cache,
            ttl_seconds=config.CACHE_TTL_SECONDS,
            max_workers=min(8, int(gdelt_workers)),
            progress_cb=x_progress,
        )

    # 6) Final ranking
    set_progress(88, "Final ranking (add X sentiment)...")
    final_df = stage1.merge(x_df, on="ticker", how="left")

    ranked = rank_final(
        final_df,
        w_x_sent=0.35,
        w_x_buzz=0.10,
    )

    # 7) Cleanup + display
    ranked["price_source"] = ranked["ticker"].apply(lambda x: src_map.get(f"{x}.JK", "none"))

    ranked["gnews_sent"] = pd.to_numeric(ranked.get("gnews_sent", np.nan), errors="coerce").round(4)
    ranked["gnews_buzz"] = pd.to_numeric(ranked.get("gnews_buzz", 0), errors="coerce")
    ranked["x_sent"] = pd.to_numeric(ranked.get("x_sent", np.nan), errors="coerce").round(4)
    ranked["x_buzz"] = pd.to_numeric(ranked.get("x_buzz", 0), errors="coerce")

    ranked["kw_used_short"] = (
        ranked.get("kw_used", "")
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 80)
    )
    ranked.loc[ranked.get("kw_used", "").fillna("").astype(str).str.len() > 80, "kw_used_short"] = (
        ranked["kw_used_short"] + "..."
    )

    set_progress(92, "Rendering results...")

    st.subheader(f"Top {top_n} candidates")
    if gnews_errors:
        st.warning("\n".join(gnews_errors))
    if advanced_mode:
        show_cols = [
            "ticker",
            "price_source",
            "last_close",
            "ret_5d",
            "avg_vol_20d",
            "gnews_sent",
            "gnews_buzz",
            "x_sent",
            "x_buzz",
            "kw_used_short",
            "score",
        ]
    else:
        show_cols = [
            "ticker",
            "last_close",
            "ret_5d",
            "avg_vol_20d",
            "gnews_sent",
            "gnews_buzz",
            "x_sent",
            "x_buzz",
            "score",
        ]

    st.dataframe(
        ranked[show_cols].head(int(top_n)),
        use_container_width=True,
    )

    # 8) Headlines (from GNews)
    if show_headlines:
        st.subheader("Headlines (GNews)")
        n_rows = int(min(len(ranked), int(top_n)))

        for i, row in ranked.head(int(top_n)).iterrows():
            t = str(row["ticker"]).strip()
            q = str(row.get("kw_used", "")).strip()

            pct = 92 + int(8 * ((i + 1) / max(1, n_rows)))  # 92..100
            set_progress(pct, f"Fetching headlines {i+1}/{n_rows} ({t})...")

            query = q if q else f'("{t}") AND (stock OR shares OR IDX OR market OR earnings)'

            headlines = fetch_latest_headlines(
                ticker=t,
                query=query,
                api_key=GNEWS_API_KEY,
                lookback_days=int(lookback_news),
                max_records=10,
                cache=cache,
                ttl_seconds=config.CACHE_TTL_SECONDS,
                lang="en",
                errors=gnews_errors,
            )

            if (headlines is None or headlines.empty) and advanced_mode:
                gnews_errors.append(f"No GNews headlines for {t}; trying GDELT fallback.")
                gdelt_query = build_gdelt_query(t, config.ALIASES)
                headlines = fetch_gdelt_headlines(
                    query=gdelt_query,
                    lookback_days_news=int(lookback_news),
                    max_records=10,
                    cache=cache,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )

            with st.expander(f"{t}", expanded=False):
                if advanced_mode:
                    st.caption("Query used:")
                    st.code(query)

                if headlines is None or headlines.empty:
                    st.write("No headlines returned.")
                    continue

                for _, h in headlines.iterrows():
                    title = str(h.get("title", "")).strip()
                    url = str(h.get("url", "")).strip()
                    src = str(h.get("source_name", h.get("domain", ""))).strip()
                    dt = str(h.get("publishedAt", h.get("seendate", ""))).strip()

                    if title and url:
                        st.markdown(f"- [{title}]({url})  \n  {src} | {dt}")
                    elif title:
                        st.markdown(f"- {title}")

    set_progress(100, "Done.")

