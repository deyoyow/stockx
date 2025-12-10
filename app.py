from __future__ import annotations

import os
from pathlib import Path
import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import config
from cache_disk import DiskCache
from data_gnews import fetch_latest_headlines
from data_gdelt import (
    build_keyword_query as build_gdelt_query,
    fetch_gdelt_metrics,
    fetch_latest_headlines as fetch_gdelt_headlines,
)
from data_x import build_x_query, fetch_x_posts
from data_yahoo import fetch_prices_fast
from scoring import (
    compute_price_features,
    fetch_gnews_for_universe,
    fetch_x_for_tickers,
    rank_final,
    rank_stage1,
)

# -------------------------------------------------------------------
# Environment + paths
# -------------------------------------------------------------------
load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "").strip()
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()

ROOT_DIR = Path(__file__).resolve().parent
LOCAL_PRICE_CSV_PATH = ROOT_DIR / "data" / "prices_manual.csv"
PORTFOLIO_CSV_PATH = ROOT_DIR / "data" / "portfolio.csv"

st.set_page_config(page_title="IDX Sentiment Screener", layout="wide")

cache = DiskCache(ROOT_DIR / ".cache")

st.title("IDX Sentiment Screener (GDELT + local prices)")
st.caption(
    "Main screener reads prices from a local CSV (scraped/offline). "
    "Use the 'Prices & Scraper' tab to update that CSV."
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_prices_from_csv(
    path: Path, tickers_jk: list[str], lookback_days_price: int
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load long-format price data from a CSV file with at least:
      ticker,date,close[,volume]

    Returns (price_df, source_map) where source_map[ticker_jk] = "csv"|"none".
    """
    tickers_jk = [t.strip() for t in tickers_jk if t and str(t).strip()]
    src_map: dict[str, str] = {t: "none" for t in tickers_jk}
    if not tickers_jk:
        return pd.DataFrame(), src_map

    if not path.exists():
        return pd.DataFrame(), src_map

    try:
        df_raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(), src_map

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), src_map

    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"ticker", "date", "close"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(), src_map

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if tickers_jk:
        df = df[df["ticker"].isin(tickers_jk)]

    if df.empty:
        return pd.DataFrame(), src_map

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=int(lookback_days_price) + 3)
    df = df[df["date"] >= cutoff]

    if df.empty:
        return pd.DataFrame(), src_map

    got = set(df["ticker"].dropna().astype(str).tolist())
    for t in tickers_jk:
        if t in got:
            src_map[t] = "csv"

    return df, src_map


def load_latest_prices(path: Path) -> pd.DataFrame:
    """
    Helper for the Portfolio tab.
    Returns one row per ticker with latest close: columns ticker,date,close.
    """
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "date", "close"])

    try:
        df_raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["ticker", "date", "close"])

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["ticker", "date", "close"])

    df = df_raw.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"ticker", "date", "close"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=["ticker", "date", "close"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date", "close"])

    df = df.sort_values(["ticker", "date"])
    latest = df.groupby("ticker").tail(1).reset_index(drop=True)
    return latest[["ticker", "date", "close"]]


# -------------------------------------------------------------------
# Sidebar controls (shared across tabs)
# -------------------------------------------------------------------
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

    st.subheader("Prices for screener")
    price_mode_ui = st.selectbox(
        "Price source",
        [
            "Manual CSV (scraped/offline)",
            "Skip prices (sentiment-only)",
        ],
        index=0,
    )

    st.subheader("Ranking")
    top_n = st.number_input(
        "Top N",
        min_value=3,
        max_value=30,
        value=config.TOP_N_DEFAULT,
        step=1,
    )
    min_vol = st.number_input(
        "Min avg daily volume (shares)",
        min_value=0,
        value=config.MIN_AVG_DAILY_VOLUME_DEFAULT,
        step=500_000,
    )

    tone_only = st.checkbox("Tone only (skip X stage)", value=False)
    show_headlines = st.checkbox("Show headlines (slower)", value=True)
    show_x_posts = st.checkbox("Show X posts (slower)", value=False)

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
            help=(
                "Only the most liquid tickers will hit GNews to avoid 429 throttling; "
                "others fall back to GDELT."
            ),
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
        gdelt_workers = st.slider(
            "GDELT workers", 2, 20, config.GDELT_WORKERS_DEFAULT, 1
        )

        st.subheader("Weights")
        w_tone = st.slider(
            "Tone weight", 0.0, 1.0, config.WEIGHT_TONE_DEFAULT, 0.05
        )
        w_vol = st.slider(
            "News volume weight", 0.0, 1.0, config.WEIGHT_NEWS_VOL_DEFAULT, 0.05
        )
        w_mom = st.slider(
            "Momentum weight", 0.0, 1.0, config.WEIGHT_MOMENTUM_DEFAULT, 0.05
        )
    else:
        prefilter_n = config.PREFILTER_N_DEFAULT
        gnews_cap = config.GNEWS_MAX_TICKERS_DEFAULT
        lookback_news = config.LOOKBACK_DAYS_NEWS_DEFAULT
        lookback_price = config.LOOKBACK_DAYS_PRICE_DEFAULT
        gdelt_workers = config.GDELT_WORKERS_DEFAULT
        w_tone = config.WEIGHT_TONE_DEFAULT
        w_vol = config.WEIGHT_NEWS_VOL_DEFAULT
        w_mom = config.WEIGHT_MOMENTUM_DEFAULT

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_screener, tab_scraper, tab_portfolio = st.tabs(
    ["Screener", "Prices & Scraper", "Portfolio / P&L"]
)

# -------------------------------------------------------------------
# Tab 1: Screener
# -------------------------------------------------------------------
with tab_screener:
    # Universe input
    universe_text = st.text_area(
        "Universe (IDX tickers, one per line, no .JK)",
        value="\n".join(config.WATCHLIST),
        height=180,
    )
    tickers = [t.strip().upper() for t in universe_text.splitlines() if t.strip()]
    tickers_jk = [f"{t}.JK" for t in tickers]

    run = st.button("Run screening", type="primary")

    if run:
        # Progress UI
        pb_slot = st.empty()
        pbar = pb_slot.progress(0, text="Starting...")

        def set_progress(pct: int, text: str) -> None:
            pct = int(max(0, min(100, pct)))
            pbar.progress(pct, text=text)

        # 1) Prices
        set_progress(5, "Loading prices...")
        price_df = pd.DataFrame()
        src_map: dict[str, str] = {t: "none" for t in tickers_jk}

        if price_mode_ui == "Skip prices (sentiment-only)":
            st.info(
                "Price source is set to 'Skip prices'; running in sentiment-only mode."
            )
        else:
            price_df, src_map = load_prices_from_csv(
                LOCAL_PRICE_CSV_PATH, tickers_jk, int(lookback_price)
            )

            src_counts = pd.Series(list(src_map.values())).value_counts()
            csv_ct = int(src_counts.get("csv", 0))
            miss_ct = int(src_counts.get("none", 0))
            st.info(
                f"Prices loaded from local CSV: CSV={csv_ct}, Missing={miss_ct} "
                f"(file: {LOCAL_PRICE_CSV_PATH.name})"
            )
            if price_df is None or price_df.empty:
                st.warning(
                    "No usable price data found in the local CSV. "
                    "Run the 'Prices & Scraper' tab or switch to sentiment-only mode."
                )

        # 2) Price features (or sentiment-only fallback)
        sentiment_only_mode = (
            price_mode_ui == "Skip prices (sentiment-only)"
            or price_df is None
            or price_df.empty
        )
        if sentiment_only_mode:
            st.warning(
                "No price data available. Using sentiment-only ranking "
                "(momentum weight forced to 0)."
            )
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
        f = f[
            (f["avg_vol_20d"].isna())
            | (f["avg_vol_20d"] >= float(min_vol))
        ]
        if f.empty:
            f = feat_df.copy()

        f = (
            f.sort_values(["avg_vol_20d", "ret_5d"], ascending=[False, False])
            .head(int(prefilter_n))
            .reset_index(drop=True)
        )

        # 4) Stage-1: GNews sentiment for most-liquid prefiltered tickers
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
            st.warning(
                "GNEWS_API_KEY not set. GNews stage will return empty -> ranking will be weak."
            )
            gnews_df = pd.DataFrame(
                columns=["ticker", "kw_used", "gnews_sent", "gnews_buzz"]
            )
        else:
            if skipped_for_gnews:
                gnews_errors.append(
                    f"Skipping GNews for {len(skipped_for_gnews)} tickers beyond cap {gnews_cap}; "
                    "they will use GDELT fallback if needed."
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

            # Fallback to GDELT for tickers where GNews returned nothing
            missing_tickers = [
                str(row["ticker"])
                for _, row in gnews_df.fillna({"gnews_buzz": 0}).iterrows()
                if (row.get("gnews_buzz", 0) == 0 and pd.isna(row.get("gnews_sent")))
            ]

            if missing_tickers:
                gnews_errors.append(
                    f"Falling back to GDELT for {len(missing_tickers)} tickers with no GNews coverage."
                )

                def gdelt_progress(
                    done: int, total: int, ticker: str | None = None
                ) -> None:
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
                            gdf.loc[t] = {
                                "kw_used": "",
                                "gnews_sent": np.nan,
                                "gnews_buzz": 0,
                            }
                        if pd.isna(gdf.loc[t].get("gnews_sent")) and gdf.loc[t].get(
                            "gnews_buzz", 0
                        ) == 0:
                            gdf.loc[t, ["kw_used", "gnews_sent", "gnews_buzz"]] = [
                                r["kw_used"],
                                r["gnews_sent"],
                                r["gnews_buzz"],
                            ]
                    gnews_df = gdf.reset_index()

        merged = f.merge(gnews_df, on="ticker", how="left")

        # 5) Stage-1 ranking (GNews + momentum)
        set_progress(72, "Ranking stage-1 (GNews + momentum)...")
        stage1 = rank_stage1(
            merged,
            w_sent=float(w_tone),
            w_buzz=float(w_vol),
            w_mom=float(w_mom),
        )

        # 6) Stage-2: X sentiment (for top 10 only)
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

        # 7) Final ranking
        set_progress(88, "Final ranking (add X sentiment)...")
        final_df = stage1.merge(x_df, on="ticker", how="left")

        ranked = rank_final(
            final_df,
            w_x_sent=0.35,
            w_x_buzz=0.10,
        )

        # 8) Cleanup + display
        ranked["price_source"] = ranked["ticker"].apply(
            lambda x: src_map.get(f"{x}.JK", "none")
        )

        ranked["gnews_sent"] = pd.to_numeric(
            ranked.get("gnews_sent", np.nan), errors="coerce"
        ).round(4)
        ranked["gnews_buzz"] = pd.to_numeric(
            ranked.get("gnews_buzz", 0), errors="coerce"
        )
        ranked["x_sent"] = pd.to_numeric(
            ranked.get("x_sent", np.nan), errors="coerce"
        ).round(4)
        ranked["x_buzz"] = pd.to_numeric(
            ranked.get("x_buzz", 0), errors="coerce"
        )

        ranked["kw_used_short"] = (
            ranked.get("kw_used", "")
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.slice(0, 80)
        )
        ranked.loc[
            ranked.get("kw_used", "").fillna("").astype(str).str.len() > 80,
            "kw_used_short",
        ] = ranked["kw_used_short"] + "..."

        set_progress(92, "Rendering results...")

        st.subheader(f"Top {top_n} candidates")
        if gnews_errors:
            if advanced_mode:
                st.warning("\n".join(gnews_errors))
            else:
                st.info(
                    f"GNews/GDELT produced {len(gnews_errors)} log messages. "
                    "Enable Advanced mode to see them."
                )

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

        # 9) Headlines (GNews + optional GDELT fallback)
        if show_headlines:
            st.subheader("Headlines (GNews)")
            n_rows = int(min(len(ranked), int(top_n)))

            for i, row in ranked.head(int(top_n)).iterrows():
                t = str(row["ticker"]).strip()
                q = str(row.get("kw_used", "")).strip()

                pct = 92 + int(8 * ((i + 1) / max(1, n_rows)))  # 92..100
                set_progress(pct, f"Fetching headlines {i+1}/{n_rows} ({t})...")

                query = (
                    q
                    if q
                    else f'("{t}") AND (stock OR shares OR IDX OR market OR earnings)'
                )

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
                    gnews_errors.append(
                        f"No GNews headlines for {t}; trying GDELT fallback."
                    )
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
                        src = str(
                            h.get("source_name", h.get("domain", ""))
                        ).strip()
                        dt = str(
                            h.get("publishedAt", h.get("seendate", ""))
                        ).strip()

                        if title and url:
                            st.markdown(
                                f"- [{title}]({url})  \n  {src} | {dt}"
                            )
                        elif title:
                            st.markdown(f"- {title}")

        # 10) X posts (detail)
        if show_x_posts:
            st.subheader("X posts (detail)")
            if tone_only:
                st.info(
                    "Tone-only mode is enabled, so the X stage was skipped."
                )
            elif not X_BEARER_TOKEN:
                st.info(
                    "X_BEARER_TOKEN is not set, so X posts cannot be fetched."
                )
            else:
                n_rows_x = int(min(len(ranked), int(top_n)))
                for i, row in ranked.head(n_rows_x).iterrows():
                    t = str(row["ticker"]).strip()
                    x_query = str(row.get("x_query", "") or "").strip()
                    if not x_query:
                        x_query = build_x_query(t, config.ALIASES)

                    posts = fetch_x_posts(
                        ticker=t,
                        query=x_query,
                        bearer_token=X_BEARER_TOKEN,
                        lookback_hours=72,
                        max_results=50,
                        cache=cache,
                        ttl_seconds=config.CACHE_TTL_SECONDS,
                    )

                    with st.expander(f"{t}", expanded=False):
                        if advanced_mode:
                            st.caption("X query used:")
                            st.code(x_query)

                        agg_sent = row.get("x_sent")
                        agg_n = int(row.get("x_buzz", 0))
                        if (
                            agg_n > 0
                            and agg_sent is not None
                            and not np.isnan(agg_sent)
                        ):
                            st.caption(
                                f"Aggregate X sentiment (compound) = {agg_sent:.3f} "
                                f"from {agg_n} posts."
                            )
                        else:
                            st.caption(
                                "No aggregate X sentiment available "
                                "(ticker may not have been in the X stage)."
                            )

                        if posts is None or posts.empty:
                            st.write("No X posts returned.")
                            continue

                        for _, p in posts.iterrows():
                            dt = str(p.get("created_at", "")).strip()
                            text = (
                                str(p.get("text", "")).strip().replace("\n", " ")
                            )
                            likes = int(p.get("like_count", 0))
                            rts = int(p.get("retweet_count", 0))
                            replies = int(p.get("reply_count", 0))
                            quotes = int(p.get("quote_count", 0))

                            st.markdown(
                                f"- {text}  \n  {dt} | "
                                f"likes={likes}, rts={rts}, replies={replies}, quotes={quotes}"
                            )

        set_progress(100, "Done.")

# -------------------------------------------------------------------
# Tab 2: Prices & Scraper
# -------------------------------------------------------------------
with tab_scraper:
    st.subheader("Update local price CSV")
    st.markdown(
        "This tab fetches prices (currently via `yfinance` through `data_yahoo.fetch_prices_fast`) "
        "and overwrites the local CSV used by the screener."
    )

    scrape_universe_text = st.text_area(
        "Tickers to scrape (IDX, one per line, no .JK)",
        value="\n".join(config.WATCHLIST),
        height=180,
    )
    scrape_tickers = [
        t.strip().upper() for t in scrape_universe_text.splitlines() if t.strip()
    ]
    scrape_tickers_jk = [f"{t}.JK" for t in scrape_tickers]

    scrape_lookback = st.number_input(
        "Price lookback for scraping (days)",
        min_value=10,
        max_value=365,
        value=config.LOOKBACK_DAYS_PRICE_DEFAULT,
        step=5,
    )

    run_scrape = st.button("Run scrape & overwrite prices CSV", type="primary")

    if run_scrape:
        if not scrape_tickers_jk:
            st.warning("Please enter at least one ticker.")
        else:
            with st.spinner("Scraping prices via yfinance..."):
                df = fetch_prices_fast(
                    tickers_jk=scrape_tickers_jk,
                    lookback_days_price=int(scrape_lookback),
                    cache=cache,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                )

            if df is None or df.empty:
                st.error("No price data returned from yfinance.")
            else:
                df2 = df.copy()
                df2.columns = [c.lower() for c in df2.columns]

                # Normalise schema for the screener
                if "date" not in df2.columns and "Date".lower() in df2.columns:
                    # already handled by lower() above
                    pass
                df2["date"] = pd.to_datetime(
                    df2.get("date"), errors="coerce"
                )
                df2 = df2.dropna(subset=["date"])

                if "volume" not in df2.columns:
                    df2["volume"] = np.nan

                out = df2[["ticker", "date", "close", "volume"]].copy()

                LOCAL_PRICE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(LOCAL_PRICE_CSV_PATH, index=False)

                st.success(
                    f"Saved {len(out)} rows for {out['ticker'].nunique()} tickers "
                    f"to {LOCAL_PRICE_CSV_PATH}"
                )
                st.dataframe(out.tail(20), use_container_width=True)

    if LOCAL_PRICE_CSV_PATH.exists():
        st.markdown("**Current local price CSV (preview):**")
        try:
            preview = pd.read_csv(LOCAL_PRICE_CSV_PATH)
            st.dataframe(preview.tail(20), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not read existing CSV: {e}")

# -------------------------------------------------------------------
# Tab 3: Portfolio / P&L
# -------------------------------------------------------------------
with tab_portfolio:
    st.subheader("Portfolio and unrealised P&L")
    st.markdown(
        "Edit your holdings below and click **Save portfolio**. "
        "P&L is computed using the latest close from the local prices CSV."
    )

    if PORTFOLIO_CSV_PATH.exists():
        try:
            port_raw = pd.read_csv(PORTFOLIO_CSV_PATH)
        except Exception:
            port_raw = pd.DataFrame(
                {"ticker": [], "quantity": [], "avg_buy_price": []}
            )
    else:
        port_raw = pd.DataFrame(
            {"ticker": [], "quantity": [], "avg_buy_price": []}
        )

    if port_raw.empty:
        port_raw = pd.DataFrame(
            {"ticker": ["BBRI", "TLKM"], "quantity": [100, 200], "avg_buy_price": [5000, 4000]}
        )

    edited_port = st.data_editor(
        port_raw,
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor",
    )

    if st.button("Save portfolio"):
        PORTFOLIO_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        edited_port.to_csv(PORTFOLIO_CSV_PATH, index=False)
        st.success(f"Portfolio saved to {PORTFOLIO_CSV_PATH}")

    # P&L calculation
    latest_prices = load_latest_prices(LOCAL_PRICE_CSV_PATH)

    if latest_prices.empty:
        st.info(
            "No price data available. Run the 'Prices & Scraper' tab first "
            "to populate the local prices CSV."
        )
    else:
        dfp = edited_port.copy()
        dfp["ticker"] = dfp.get("ticker", "").astype(str).str.strip().str.upper()
        dfp["ticker_jk"] = dfp["ticker"].apply(
            lambda x: x if x.endswith(".JK") else (f"{x}.JK" if x else "")
        )

        dfp["quantity"] = pd.to_numeric(
            dfp.get("quantity", 0.0), errors="coerce"
        )
        dfp["avg_buy_price"] = pd.to_numeric(
            dfp.get("avg_buy_price", 0.0), errors="coerce"
        )

        latest = latest_prices.copy()
        latest["ticker_jk"] = latest["ticker"].astype(str).str.upper()

        merged = dfp.merge(
            latest[["ticker_jk", "date", "close"]],
            on="ticker_jk",
            how="left",
        )

        merged["last_price"] = pd.to_numeric(
            merged.get("close", np.nan), errors="coerce"
        )
        merged["cost_basis"] = merged["quantity"] * merged["avg_buy_price"]
        merged["market_value"] = merged["quantity"] * merged["last_price"]
        merged["pnl"] = merged["market_value"] - merged["cost_basis"]
        merged["pnl_pct"] = np.where(
            merged["cost_basis"] > 0,
            100.0 * merged["pnl"] / merged["cost_basis"],
            np.nan,
        )

        # Summary metrics
        valid = merged.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["quantity", "avg_buy_price"], how="any"
        )
        total_cost = float(valid["cost_basis"].sum())
        total_value = float(valid["market_value"].sum())
        total_pnl = total_value - total_cost
        total_pnl_pct = (
            100.0 * total_pnl / total_cost if total_cost > 0 else np.nan
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total cost", f"{total_cost:,.0f}")
        c2.metric("Total value", f"{total_value:,.0f}")
        if not np.isnan(total_pnl_pct):
            c3.metric(
                "Unrealised P&L",
                f"{total_pnl:,.0f} ({total_pnl_pct:+.2f}%)",
            )
        else:
            c3.metric("Unrealised P&L", f"{total_pnl:,.0f}")

        show_cols = [
            "ticker",
            "quantity",
            "avg_buy_price",
            "last_price",
            "cost_basis",
            "market_value",
            "pnl",
            "pnl_pct",
        ]
        merged["ticker"] = merged["ticker"].fillna(merged["ticker_jk"])
        st.dataframe(
            merged[show_cols],
            use_container_width=True,
        )
