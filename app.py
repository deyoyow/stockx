from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import config
from cache_disk import DiskCache
from data_gdelt import (
    build_keyword_query as build_gdelt_query,
    fetch_gdelt_metrics,
    fetch_latest_headlines as fetch_gdelt_headlines,
)
from data_gnews import fetch_latest_headlines
from data_prices import load_prices_from_csv
from data_x import build_x_query, fetch_x_posts
from scoring import (
    compute_price_features,
    fetch_gnews_for_universe,
    fetch_x_for_tickers,
    rank_final,
    rank_stage1,
)
from data_idx import fetch_idx_prices


ROOT_DIR = Path(__file__).resolve().parent
LOCAL_PRICE_CSV_PATH = ROOT_DIR / "data" / "price_idx.csv"
PORTFOLIO_CSV_PATH = ROOT_DIR / "data" / "portfolio.csv"

ROOT_DIR = Path(__file__).resolve().parent
LOCAL_PRICE_CSV_PATH = ROOT_DIR / "data" / "price_idx.csv"
PORTFOLIO_CSV_PATH = ROOT_DIR / "data" / "portfolio.csv"

ROOT_DIR = Path(__file__).resolve().parent
LOCAL_PRICE_CSV_PATH = ROOT_DIR / "data" / "price_idx.csv"
PORTFOLIO_CSV_PATH = ROOT_DIR / "data" / "portfolio.csv"

load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "").strip()
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()


st.set_page_config(page_title="IDX Sentiment Screener", layout="wide")

cache = DiskCache(ROOT_DIR / ".cache")

st.title("IDX Sentiment Screener")
st.caption("News-driven ranking with optional CSV prices and disk caching.")




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

    st.subheader("Price source (screener)")
    price_source = st.selectbox(
        "Price data",
        [
            "From CSV (offline)",
            "Skip prices (sentiment-only)",
        ],
        index=0,
        help="Read cached CSV prices or skip prices entirely.",
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


def _parse_ticker_text(val: str) -> list[str]:
    return [t.strip().upper() for t in val.splitlines() if t.strip()]


def _build_src_map(tickers_jk: list[str], present: set[str], label: str) -> dict[str, str]:
    src_map = {t: "none" for t in tickers_jk}
    for t in tickers_jk:
        if t in present:
            src_map[t] = label
    return src_map


def _filter_prices_by_lookback(df: pd.DataFrame, lookback_days_price: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    start = pd.Timestamp.utcnow() - pd.Timedelta(days=int(lookback_days_price) + 3)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[df["date"] >= start]


tab_screener, tab_scraper, tab_portfolio = st.tabs([
    "Screener",
    "Prices & Scraper",
    "Portfolio / P&L",
])


# ----------------------------
# Screener tab
# ----------------------------
with tab_screener:
    st.subheader("IDX Sentiment Screener")
    universe_text = st.text_area(
        "Universe (IDX tickers, one per line, no .JK)",
        value="\n".join(config.WATCHLIST),
        height=180,
    )

    tickers = _parse_ticker_text(universe_text)
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
        set_progress(5, "Fetching prices...")
        price_df = pd.DataFrame()
        src_map: dict[str, str] = {t: "none" for t in tickers_jk}
        price_error: str | None = None

        if price_source == "Skip prices (sentiment-only)":
            sentiment_only_mode = True
        else:
            price_df, src_map = load_prices_from_csv(
                path=LOCAL_PRICE_CSV_PATH,
                tickers_jk=tickers_jk,
                lookback_days_price=int(lookback_price),
            )
            sentiment_only_mode = price_df.empty
            if sentiment_only_mode:
                price_error = (
                    "No usable price data in price_idx.csv. Run `python update_prices_yahoo.py` "
                    "to refresh prices or switch to sentiment-only."
                )

        src_counts = pd.Series(list(src_map.values())).value_counts() if src_map else pd.Series()
        st.info(
            "Prices loaded: "
            + ", ".join(
                [f"{k}={int(v)}" for k, v in src_counts.to_dict().items()]
            )
        )
        if price_error:
            st.warning(price_error)

        # 2) Price features (or sentiment-only fallback)
        w_mom_effective = float(w_mom)
        if sentiment_only_mode:
            st.warning("Using sentiment-only ranking (momentum weight forced to 0).")
            w_mom_effective = 0.0
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

        gnews_errors: list[str] | None = [] if advanced_mode else None

        def gnews_progress(done: int, total: int, ticker: str | None = None) -> None:
            total = max(1, int(total))
            done = int(done)
            pct = 30 + int(40 * (done / total))  # 30..70
            name = f" ({ticker})" if ticker else ""
            set_progress(pct, f"GNews sentiment {done}/{total}{name}...")

        if not GNEWS_API_KEY:
            if advanced_mode:
                st.warning("GNEWS_API_KEY not set. GNews stage will return empty -> ranking will be weak.")
            gnews_df = pd.DataFrame(columns=["ticker", "kw_used", "gnews_sent", "gnews_buzz"])
        else:
            if skipped_for_gnews and gnews_errors is not None:
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

            if missing_tickers and gnews_errors is not None:
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

        # 5) Stage-1 ranking
        set_progress(72, "Ranking stage-1 (GNews + momentum)...")
        stage1 = rank_stage1(
            merged,
            w_sent=float(w_tone),
            w_buzz=float(w_vol),
            w_mom=float(w_mom_effective),
        )

        # 6) Stage-2: X sentiment for only Top 10 from stage-1
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
            st.info("Skipping X stage (tone only mode).")
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
        if advanced_mode and gnews_errors:
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
            width="stretch",
        )

        # 8) Headlines (from GNews)
        if show_headlines:
            st.subheader("Headlines")
            n_rows = int(min(len(ranked), int(top_n)))

            for i, row in ranked.head(int(top_n)).iterrows():
                t = str(row["ticker"]).strip()
                q = str(row.get("kw_used", "")).strip()

                pct = 92 + int(8 * ((i + 1) / max(1, n_rows)))  # 92..100
                set_progress(pct, f"Fetching headlines {i+1}/{n_rows} ({t})...")

                query = q if q else f'("{t}") AND (stock OR shares OR IDX OR market OR earnings)'

                errors_list: list[str] | None = [] if advanced_mode else None
                headlines = fetch_latest_headlines(
                    ticker=t,
                    query=query,
                    api_key=GNEWS_API_KEY,
                    lookback_days=int(lookback_news),
                    max_records=10,
                    cache=cache,
                    ttl_seconds=config.CACHE_TTL_SECONDS,
                    lang="en",
                    errors=errors_list,
                )

                if (headlines is None or headlines.empty) and advanced_mode:
                    if errors_list is not None:
                        errors_list.append(f"No GNews headlines for {t}; trying GDELT fallback.")
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
                        if errors_list:
                            st.warning("\n".join(errors_list))

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

        # 9) Optional X posts view
        posts_expander = st.expander("Show X posts (slower)", expanded=False)
        with posts_expander:
            if tone_only:
                st.info("X stage skipped (tone only mode).")
            elif not X_BEARER_TOKEN:
                st.info("Set X_BEARER_TOKEN to fetch recent posts.")
            else:
                for t in ranked.head(int(top_n))["ticker"].tolist():
                    q = build_x_query(t, config.ALIASES)
                    st.markdown(f"**{t}** — query: `{q}`")
                    try:
                        posts = fetch_x_posts(
                            ticker=t,
                            query=q,
                            bearer_token=X_BEARER_TOKEN,
                            lookback_hours=72,
                            max_results=20,
                            cache=cache,
                            ttl_seconds=config.CACHE_TTL_SECONDS,
                        )
                    except Exception as e:  # pragma: no cover - defensive UI guard
                        st.error(f"Failed to load posts for {t}: {e}")
                        continue

                    if posts is None or posts.empty:
                        st.write("No recent posts.")
                        continue

                    for _, p in posts.sort_values("created_at", ascending=False).iterrows():
                        text = str(p.get("text", "")).strip()
                        ts = p.get("created_at")
                        st.markdown(f"- {text}\n  {ts}")

        set_progress(100, "Done.")


# ----------------------------
# Prices & Scraper tab
# ----------------------------
with tab_scraper:
    st.subheader("Prices & Scraper (offline updater)")
    st.write(
        "Prices are loaded from `data/price_idx.csv`. Use the CLI script to refresh "
        "prices without hammering Yahoo:"
    )
    st.code("python update_prices_yahoo.py --lookback 180")

    st.write("Watchlist tickers used by the CLI:")
    st.text("\n".join(config.WATCHLIST))

    prices_path = LOCAL_PRICE_CSV_PATH
    if prices_path.exists():
        st.markdown("### Current price_idx.csv (last 20 rows)")
        try:
            preview = pd.read_csv(prices_path)
            st.dataframe(preview.tail(20), width="stretch")
        except Exception as e:  # pragma: no cover - UI safety
            st.error(f"Failed to read existing CSV: {e}")
    else:
        st.info("price_idx.csv not found yet. Run `python update_prices_yahoo.py` to generate it.")


# ----------------------------
# Portfolio tab
# ----------------------------
with tab_portfolio:
    st.subheader("Portfolio / P&L")
    port_path = PORTFOLIO_CSV_PATH

    if port_path.exists():
        try:
            portfolio_df = pd.read_csv(port_path)
        except Exception:
            portfolio_df = pd.DataFrame(columns=["ticker", "quantity", "avg_buy_price"])
    else:
        portfolio_df = pd.DataFrame(columns=["ticker", "quantity", "avg_buy_price"])

    if portfolio_df.empty:
        portfolio_df = pd.DataFrame({
            "ticker": [],
            "quantity": [],
            "avg_buy_price": [],
        })

    st.write("Edit holdings (ticker without .JK). Use 'Save portfolio' to persist changes.")
    edited = st.data_editor(
        portfolio_df,
        num_rows="dynamic",
        width="stretch",
        key="portfolio_editor",
    )

    if st.button("Save portfolio"):
        try:
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            edited.to_csv(port_path, index=False)
            st.success(f"Saved portfolio to {port_path}.")
        except Exception as e:  # pragma: no cover
            st.error(f"Failed to save portfolio: {e}")

    if edited is not None and not edited.empty:
        try:
            edited["ticker"] = edited["ticker"].astype(str).str.upper().str.strip()
            edited["quantity"] = pd.to_numeric(edited.get("quantity", 0), errors="coerce").fillna(0.0)
            edited["avg_buy_price"] = pd.to_numeric(edited.get("avg_buy_price", 0), errors="coerce").fillna(0.0)
        except Exception:
            st.error("Invalid data in portfolio; please fix values.")
        else:
            tickers_needed = [f"{t}.JK" for t in edited["ticker"].dropna().astype(str).tolist() if t]
            prices_df, _ = load_prices_from_csv(LOCAL_PRICE_CSV_PATH, tickers_needed, lookback_days_price=365)
            last_prices = pd.Series(dtype=float)
            if not prices_df.empty:
                last_prices = (
                    prices_df.sort_values("date")
                    .groupby("ticker")
                    .last()["close"]
                )

            def _lookup_price(ticker_base: str) -> float:
                jk = f"{ticker_base}.JK"
                return float(last_prices.get(jk, np.nan)) if not last_prices.empty else np.nan

            df_calc = edited.copy()
            df_calc["last_price"] = df_calc["ticker"].apply(_lookup_price)
            df_calc["cost_basis"] = df_calc["quantity"] * df_calc["avg_buy_price"]
            df_calc["market_value"] = df_calc["quantity"] * df_calc["last_price"]
            df_calc["pnl"] = df_calc["market_value"] - df_calc["cost_basis"]
            df_calc["pnl_pct"] = np.where(
                df_calc["cost_basis"] > 0,
                (df_calc["pnl"] / df_calc["cost_basis"]) * 100,
                np.nan,
            )

            total_cost = df_calc["cost_basis"].sum()
            total_mv = df_calc["market_value"].sum()
            total_pnl = df_calc["pnl"].sum()
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total cost", f"{total_cost:,.0f}")
            c2.metric("Total market value", f"{total_mv:,.0f}")
            c3.metric("Total P&L", f"{total_pnl:,.0f}")
            if not np.isnan(total_pnl_pct):
                c4.metric("P&L %", f"{total_pnl_pct:,.2f}%")
            else:
                c4.metric("P&L %", "n/a")

            st.markdown("### Holdings with P&L")
            st.dataframe(
                df_calc[
                    [
                        "ticker",
                        "quantity",
                        "avg_buy_price",
                        "last_price",
                        "cost_basis",
                        "market_value",
                        "pnl",
                        "pnl_pct",
                    ]
                ],
                width="stretch",
            )
    else:
        st.info("Add holdings above to see P&L against price_idx.csv.")
