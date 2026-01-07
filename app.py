import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

API_URL = (
    "https://dlmm-api.meteora.ag/pair/all_with_pagination?"
    "limit=100&sort_key=feetvlratio1h&order_by=desc&include_unknown=true&hide_low_tvl=1000"
    "&hide_low_apr=true&include_token_mints=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v%2C%20So11111111111111111111111111111111111111112"
)
JUPITER_SEARCH_URL = "https://lite-api.jup.ag/tokens/v2/search?query="
JUPITER_ULTRA_URL = "https://lite-api.jup.ag/ultra/v1/search?query="
SOL_PRICE_URL = "https://lite-api.jup.ag/price/v3?ids=So11111111111111111111111111111111111111112"

ULTRA_TIMEFRAMES = ["stats5m", "stats1h", "stats6h", "stats24h"]

ENTRY_VOL_LIQ_5M_MIN = 0.3
EXIT_RATIO_DROP_MULTIPLIER = 0.90
COOLDOWN_MULTIPLIER = 0.5

EPS = 1e-12


# ----------------------------
# Auto-refresh (session-safe)
# ----------------------------
def ensure_autorefresh_state(interval_sec: int) -> None:
    if "autorefresh_interval_sec" not in st.session_state:
        st.session_state.autorefresh_interval_sec = interval_sec
    if "next_refresh_ts" not in st.session_state:
        st.session_state.next_refresh_ts = time.time() + interval_sec


def autorefresh_tick() -> None:
    """
    Session-preserving auto-refresh.
    Sleeps in small increments until it's time to rerun, then calls st.rerun().
    """
    interval_sec = int(st.session_state.autorefresh_interval_sec)
    next_ts = float(st.session_state.next_refresh_ts)

    remaining = next_ts - time.time()
    if remaining <= 0:
        st.session_state.next_refresh_ts = time.time() + interval_sec
        st.rerun()

    # Show countdown and wait a bit. This keeps session_state intact.
    # (We use a small sleep so the page doesn't freeze for 30s without updates.)
    placeholder = st.empty()
    placeholder.caption(f"Auto-refresh in ~{int(remaining)}s (session-safe).")

    time.sleep(min(1.0, max(0.0, remaining)))
    # After sleeping 1s (or less), rerun to update countdown and check again.
    st.rerun()


# ----------------------------
# Fetching
# ----------------------------
def fetch_data(api_url: str) -> Optional[dict]:
    try:
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"API Error: {e}")
        st.warning("⚠️ Failed to fetch pool data from Meteora API.")
        return None


def fetch_jupiter_data(mint_addresses: List[str]) -> Dict[str, Dict[str, Any]]:
    if not mint_addresses:
        return {}
    query = ",".join(mint_addresses)
    try:
        response = requests.get(f"{JUPITER_SEARCH_URL}{query}", timeout=15)
        response.raise_for_status()
        data = response.json()
        return {
            item["id"]: {
                "organicScore": item.get("organicScore", 0),
                "mcap": item.get("mcap", 0),
                "usdPrice": item.get("usdPrice", 0),
                "decimals": item.get("decimals", 9),
                "createdAt": item.get("firstPool", {}).get("createdAt", ""),
                "liquidity": item.get("liquidity", 0),
            }
            for item in data
        }
    except Exception as e:
        logging.error(f"Jupiter API Error: {e}")
        st.warning("⚠️ Failed to fetch Jupiter token data.")
        return {}


def fetch_ultra_stats(mint_addresses: List[str]) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    if not mint_addresses:
        return stats
    query = ",".join(mint_addresses[:100])
    try:
        response = requests.get(f"{JUPITER_ULTRA_URL}{query}", timeout=20)
        response.raise_for_status()
        data = response.json()
        for item in data:
            mint = item.get("id")
            pool_stats = {tf: item.get(tf, {}) for tf in ULTRA_TIMEFRAMES}
            pool_stats["ultra_liquidity"] = item.get("liquidity", 0)
            stats[mint] = pool_stats
        return stats
    except Exception as e:
        logging.error(f"Ultra API Error: {e}")
        st.warning("⚠️ Failed to fetch Jupiter Ultra stats.")
        return {}


def fetch_sol_price() -> float:
    try:
        response = requests.get(SOL_PRICE_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("So11111111111111111111111111111111111111112", {}).get("usdPrice", 0)
    except Exception as e:
        logging.error(f"SOL price API Error: {e}")
        st.warning("⚠️ Failed to fetch SOL price.")
        return 0.0


# ----------------------------
# Time / parsing helpers
# ----------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def format_created_at(created_at_str: str) -> str:
    try:
        created_time = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - created_time
        days = diff.days
        hours = diff.seconds // 3600
        if days >= 1:
            return f"{days} day{'s' if days != 1 else ''} {hours} hours"
        return f"{hours} hours"
    except Exception:
        return "N/A"


def parse_created_to_hours(created_str: str) -> int:
    try:
        if created_str == "N/A":
            return 0
        parts = created_str.split()
        if len(parts) >= 4 and "day" in parts[1]:
            days = int(parts[0])
            hours = int(parts[2])
            return days * 24 + hours
        if len(parts) >= 2 and "hour" in parts[1]:
            return int(parts[0])
        return 0
    except Exception:
        return 0


# ----------------------------
# Pair processing helpers
# ----------------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def get_true_averages(p: dict, key: str) -> float:
    timeframes = [1, 2, 4, 12, 24]
    values = [float(p.get(key, {}).get(f"hour_{tf}", 0)) for tf in timeframes]

    filtered: List[float] = []
    last_val: Optional[float] = None
    for val in values:
        if last_val is None or val != last_val:
            filtered.append(val)
        last_val = val

    if filtered:
        if all(v == 0 for v in filtered):
            return 0.0
        return sum(filtered) / len(filtered)
    return 0.0


def compute_entry_score(vol_liq_5m: float, vol_liq_1h: float, ratio_min30: float) -> float:
    heat = clamp((vol_liq_5m - 0.3) / 0.7, 0.0, 1.0)
    accel_ratio = vol_liq_5m / (vol_liq_1h + EPS)
    accel = clamp((accel_ratio - 1.0) / 1.0, 0.0, 1.0)
    fee = clamp((ratio_min30 - 2.0) / 8.0, 0.0, 1.0)
    return float(100.0 * (0.50 * heat + 0.35 * accel + 0.15 * fee))


def compute_exit_score(
    vol_liq_5m: float,
    vol_liq_1h: float,
    ratio_min30: float,
    entry_ratio_min30: float,
) -> float:
    fee_drop = clamp((entry_ratio_min30 - ratio_min30) / (entry_ratio_min30 + EPS), 0.0, 1.0)
    target = COOLDOWN_MULTIPLIER * vol_liq_1h
    cool = clamp((target - vol_liq_5m) / (target + EPS), 0.0, 1.0)
    return float(100.0 * (0.65 * fee_drop + 0.35 * cool))


def process_pairs(
    raw_pairs: List[dict],
    jupiter_data: Dict[str, dict],
    sol_price: float,
    ultra_stats: Dict[str, dict],
) -> List[dict]:
    processed: List[dict] = []
    for p in raw_pairs:
        mint_x = p.get("mint_x", "N/A")
        mint_y = p.get("mint_y", "N/A")

        extra = jupiter_data.get(mint_x, {})
        if extra.get("organicScore", 0) < 74.9:
            continue

        address = p.get("address", "N/A")
        meteora_link = f"<a href='https://app.meteora.ag/dlmm/{address}' target='_blank' style='white-space:nowrap;'>meteora</a>"
        dexscreener_link = f"<a href='https://dexscreener.com/solana/{mint_x}' target='_blank' style='white-space:nowrap;'>dex</a>"
        jupiter_link = f"<a href='https://jup.ag/swap?sell=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v&buy={mint_x}' target='_blank' style='white-space:nowrap;'>jupiter</a>"
        combined_links = f"<div style='text-align:center;line-height:1.2;'>{meteora_link}<br>{dexscreener_link}<br>{jupiter_link}</div>"

        decimals_x = extra.get("decimals", 9)
        reserve_x = float(p.get("reserve_x_amount", 0)) / (10**decimals_x)

        if mint_y == "So11111111111111111111111111111111111111112":
            decimals_y = 9
            price_y = sol_price
        else:
            decimals_y = 6
            price_y = 1

        reserve_y = float(p.get("reserve_y_amount", 0)) / (10**decimals_y)
        price_x = extra.get("usdPrice", 0)

        usd_x = reserve_x * price_x
        usd_y = reserve_y * price_y
        total = usd_x + usd_y
        percent_x = (usd_x / total * 100) if total > 0 else 0

        created_str = format_created_at(extra.get("createdAt", ""))

        avg_vol = get_true_averages(p, "volume")
        avg_fee = get_true_averages(p, "fees")
        avg_ratio = get_true_averages(p, "fee_tvl_ratio")

        vol_min30 = float(p.get("volume", {}).get("min_30", 0))
        ratio_min30 = float(p.get("fee_tvl_ratio", {}).get("min_30", 0))
        mcap = extra.get("mcap", 0)
        custom_sort = vol_min30 * ratio_min30 * mcap if ratio_min30 >= 2 else None

        ultra = ultra_stats.get(mint_x, {})
        ultra_liquidity = float(ultra.get("ultra_liquidity", 0))

        vol_liq_dict: Dict[str, float] = {}
        for tf in ULTRA_TIMEFRAMES:
            tf_stats = ultra.get(tf, {})
            buy_vol = float(tf_stats.get("buyVolume", 0))
            sell_vol = float(tf_stats.get("sellVolume", 0))
            total_vol = buy_vol + sell_vol
            vol_liq = (total_vol / ultra_liquidity) if ultra_liquidity > 0 else 0.0
            vol_liq_dict[tf.replace("stats", "vol/liq ")] = vol_liq

        item: Dict[str, Any] = {
            "Links": combined_links,
            "Name": p.get("name", "N/A"),
            "Liquidity ($)": float(p.get("liquidity", 0)),
            "Vol min30": vol_min30,
            "Fee min30": float(p.get("fees", {}).get("min_30", 0)),
            "Ratio min30": ratio_min30,
            "Bin Step": p.get("bin_step", "N/A"),
            "Base Fee %": float(p.get("base_fee_percentage", 0)),
            "Max Fee %": float(p.get("max_fee_percentage", 0)),
            "Created": created_str,
            "MCAP": mcap,
            "Organic Score": extra.get("organicScore", 0),
            "LP Ratio": percent_x,
            "Avg Vol": avg_vol,
            "Avg Fee": avg_fee,
            "Avg Ratio": avg_ratio,
            "Vol 1h": float(p.get("volume", {}).get("hour_1", 0)),
            "Fee 1h": float(p.get("fees", {}).get("hour_1", 0)),
            "Ratio 1h": float(p.get("fee_tvl_ratio", {}).get("hour_1", 0)),
            "Vol 2h": float(p.get("volume", {}).get("hour_2", 0)),
            "Fee 2h": float(p.get("fees", {}).get("hour_2", 0)),
            "Ratio 2h": float(p.get("fee_tvl_ratio", {}).get("hour_2", 0)),
            "Vol 4h": float(p.get("volume", {}).get("hour_4", 0)),
            "Fee 4h": float(p.get("fees", {}).get("hour_4", 0)),
            "Ratio 4h": float(p.get("fee_tvl_ratio", {}).get("hour_4", 0)),
            "Vol 12h": float(p.get("volume", {}).get("hour_12", 0)),
            "Fee 12h": float(p.get("fees", {}).get("hour_12", 0)),
            "Ratio 12h": float(p.get("fee_tvl_ratio", {}).get("hour_12", 0)),
            "Vol 24h": float(p.get("volume", {}).get("hour_24", 0)),
            "Fee 24h": float(p.get("fees", {}).get("hour_24", 0)),
            "Ratio 24h": float(p.get("fee_tvl_ratio", {}).get("hour_24", 0)),
            "Fees 24h ($)": float(p.get("fees_24h", 0)),
            "Today Fees ($)": float(p.get("today_fees", 0)),
            "Trade Vol 24h ($)": float(p.get("trade_volume_24h", 0)),
            "Cum Trade Vol": float(p.get("cumulative_trade_volume", 0)),
            "Cum Fee Vol": float(p.get("cumulative_fee_volume", 0)),
            "Custom Sort": custom_sort,
            "Address": address,
        }
        item.update(vol_liq_dict)
        processed.append(item)

    return processed


def get_safe_default(min_val: float, max_val: float, values_list: List[float]) -> float:
    if min_val > 0:
        return min_val
    non_zero_values = sorted([v for v in values_list if v > 0])
    if non_zero_values:
        return non_zero_values[0]
    return max_val


def safe_session_number(key: str, min_val: float, max_val: float, default: float) -> float:
    if key not in st.session_state or not isinstance(st.session_state[key], (int, float)):
        st.session_state[key] = default
    else:
        if st.session_state[key] < min_val:
            st.session_state[key] = min_val
        elif st.session_state[key] > max_val:
            st.session_state[key] = max_val
    return st.session_state[key]


def init_positions_state() -> None:
    if "positions" not in st.session_state or not isinstance(st.session_state.positions, dict):
        st.session_state.positions = {}


def parse_positions_text(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    return [ln for ln in lines if ln]


def compute_signals(pairs: List[dict], positions: Dict[str, Dict[str, Any]]) -> List[dict]:
    enriched: List[dict] = []
    for p in pairs:
        address = p.get("Address", "")
        v5 = float(p.get("vol/liq 5m", 0) or 0)
        v1 = float(p.get("vol/liq 1h", 0) or 0)
        r30 = float(p.get("Ratio min30", 0) or 0)

        entry_score = compute_entry_score(v5, v1, r30)

        in_pos = address in positions
        entry_ratio = float(positions.get(address, {}).get("entry_ratio_min30", 0) or 0)
        entered_at = positions.get(address, {}).get("entered_at", "")

        ratio_exit = (entry_ratio > 0) and (r30 < entry_ratio * EXIT_RATIO_DROP_MULTIPLIER)
        cooldown_exit = v5 < (COOLDOWN_MULTIPLIER * v1)

        exit_score = 0.0
        if in_pos and entry_ratio > 0:
            exit_score = compute_exit_score(v5, v1, r30, entry_ratio)

        entry_ok = (v5 > ENTRY_VOL_LIQ_5M_MIN) and (v5 > v1)

        if not in_pos:
            signal = "ENTER" if entry_ok else "HOLD"
        else:
            signal = "EXIT" if (ratio_exit or cooldown_exit) else "IN POSITION"

        p2 = dict(p)
        p2["Entry Score"] = entry_score
        p2["Exit Score"] = exit_score
        p2["Signal"] = signal
        p2["Entry Ratio"] = entry_ratio
        p2["Entered At"] = entered_at
        enriched.append(p2)
    return enriched


def format_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatters = {
        "Liquidity ($)": lambda x: f"{x:,.2f}",
        "Vol min30": lambda x: f"{x:,.2f}",
        "Fee min30": lambda x: f"{x:,.2f}",
        "Ratio min30": lambda x: f"{x:.2f}",
        "Avg Vol": lambda x: f"{x:,.2f}",
        "Avg Fee": lambda x: f"{x:,.2f}",
        "Avg Ratio": lambda x: f"{x:.2f}",
        "Base Fee %": lambda x: f"{x:.2f}%",
        "Max Fee %": lambda x: f"{x:.2f}%",
        "MCAP": lambda x: f"{x:,.2f}",
        "Organic Score": lambda x: f"{x:.2f}",
        "LP Ratio": lambda x: f"{x:.0f}% / {100 - float(x):.0f}%" if isinstance(x, (float, int)) else x,
        "Fees 24h ($)": lambda x: f"{x:,.2f}",
        "Today Fees ($)": lambda x: f"{x:,.2f}",
        "Trade Vol 24h ($)": lambda x: f"{x:,.2f}",
        "Cum Trade Vol": lambda x: f"{x:,.2f}",
        "Cum Fee Vol": lambda x: f"{x:,.2f}",
        "Custom Sort": lambda x: f"{x:,.0f}" if x is not None else "",
        "Entry Score": lambda x: f"{x:.1f}",
        "Exit Score": lambda x: f"{x:.1f}",
        "Entry Ratio": lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else "",
    }

    for tf in ULTRA_TIMEFRAMES:
        colname = tf.replace("stats", "vol/liq ")
        formatters[colname] = lambda x: f"{float(x):.5f}"

    for tf in ["1h", "2h", "4h", "12h", "24h"]:
        formatters[f"Vol {tf}"] = lambda x: f"{x:,.2f}"
        formatters[f"Fee {tf}"] = lambda x: f"{x:,.2f}"
        formatters[f"Ratio {tf}"] = lambda x: f"{x:.2f}"

    for col, func in formatters.items():
        if col in df.columns:
            df[col] = df[col].apply(func)

    return df


def display_table(pairs: List[dict], sort_field: str, reverse: bool) -> None:
    df = pd.DataFrame(pairs)
    df = format_columns(df)

    columns = [
        "Signal",
        "Entry Score",
        "Exit Score",
        "Entry Ratio",
        "Entered At",
        "Links",
        "Name",
        "Liquidity ($)",
        "Vol min30",
        "Fee min30",
        "Ratio min30",
        "Bin Step",
        "Base Fee %",
        "Max Fee %",
        "Created",
        "MCAP",
        "Organic Score",
        "LP Ratio",
        "Avg Vol",
        "Avg Fee",
        "Avg Ratio",
        "Vol 1h",
        "Fee 1h",
        "Ratio 1h",
        "Vol 2h",
        "Fee 2h",
        "Ratio 2h",
        "Vol 4h",
        "Fee 4h",
        "Ratio 4h",
        "Vol 12h",
        "Fee 12h",
        "Ratio 12h",
        "Vol 24h",
        "Fee 24h",
        "Ratio 24h",
        "Fees 24h ($)",
        "Today Fees ($)",
        "Trade Vol 24h ($)",
        "Cum Trade Vol",
        "Cum Fee Vol",
        "Custom Sort",
        *[tf.replace("stats", "vol/liq ") for tf in ULTRA_TIMEFRAMES],
        "Address",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = 0.0 if "vol/liq" in col else ""

    df = df[[col for col in columns if col in df.columns]]

    try:
        sort_vals = df[sort_field].astype(str).str.replace(",", "").str.replace("%", "").astype(float)
        df = (
            df.assign(_sort_col=sort_vals)
            .sort_values("_sort_col", ascending=not reverse)
            .drop("_sort_col", axis=1)
        )
    except Exception:
        df = df.sort_values(sort_field, ascending=not reverse)

    st.markdown(
        """
        <style>
        thead th {
            position: sticky;
            top: 65px;
            background-color: #222 !important;
            z-index: 2;
            text-align: center !important;
            vertical-align: middle !important;
            white-space: nowrap !important;
            padding-top: 1px !important;
            padding-bottom: 1px !important;
        }
        tbody td {
            text-align: center !important;
            vertical-align: middle !important;
            white-space: nowrap !important;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 340px;
            padding-top: 1px !important;
            padding-bottom: 1px !important;
        }
        td {
            word-break: break-all !important;
            overflow-wrap: anywhere !important;
        }
        td[data-label="Address"] {
            max-width: 500px !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-all !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def row_style(row: pd.Series) -> List[str]:
        sig = str(row.get("Signal", ""))
        if sig == "ENTER":
            return ["background-color: #0b3d0b; color: #fff;"] * len(row)
        if sig == "IN POSITION":
            return ["background-color: #6b5b00; color: #fff;"] * len(row)
        if sig == "EXIT":
            return ["background-color: #5a0b0b; color: #fff;"] * len(row)
        return [""] * len(row)

    styled_df = df.style.apply(row_style, axis=1)

    st.write("### Meteora Pools Table (Signals)")
    st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.download_button(
        label="Download table as CSV",
        data=df.to_csv(index=False),
        file_name="meteora_pools.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Meteora Pool Scoring Dashboard")
    st.write("This dashboard fetches and scores pools from Meteora, with interactive sorting, filters, and signals.")

    init_positions_state()
    ensure_autorefresh_state(interval_sec=30)

    st.caption(f"Last updated (UTC): {now_utc_iso()} | Auto-refresh: 30s (session-safe)")

    data = fetch_data(API_URL)
    sol_price = fetch_sol_price()

    if not data or "pairs" not in data:
        st.warning("No data received or 'pairs' key missing. Try refreshing, or check the Meteora API status.")
        return

    mint_x_list = [p.get("mint_x") for p in data["pairs"] if p.get("mint_x")]
    jupiter_data = fetch_jupiter_data(mint_x_list) if mint_x_list else {}
    ultra_stats = fetch_ultra_stats(mint_x_list) if mint_x_list else {}
    pairs = process_pairs(data["pairs"], jupiter_data, sol_price, ultra_stats)

    if not pairs:
        st.warning("No pairs passed the filtering (organicScore ≥ 74.9).")
        return

    created_hours = [parse_created_to_hours(p["Created"]) for p in pairs]
    mcap_values = [p["MCAP"] for p in pairs]
    vol_30mins_list = [p["Vol min30"] for p in pairs]
    ratio_min30_list = [p["Ratio min30"] for p in pairs]

    min_age, max_age = int(min(created_hours)), int(max(created_hours))
    min_mcap, max_mcap = int(min(mcap_values)), int(max(mcap_values))
    min_vol_30, max_vol_30 = float(min(vol_30mins_list)), float(max(vol_30mins_list))
    min_ratio_min30, max_ratio_min30 = float(min(ratio_min30_list)), float(max(ratio_min30_list))

    min_age_default = get_safe_default(min_age, max_age, created_hours)
    min_mcap_default = get_safe_default(min_mcap, max_mcap, mcap_values)
    min_vol_30_default = get_safe_default(min_vol_30, max_vol_30, vol_30mins_list)
    min_ratio_min30_default = get_safe_default(min_ratio_min30, max_ratio_min30, ratio_min30_list)

    if "filter_settings" not in st.session_state:
        st.session_state.filter_settings = {
            "min_age": min_age_default,
            "min_mcap": min_mcap_default,
            "min_vol_30": min_vol_30_default,
            "min_ratio_min30": min_ratio_min30_default,
        }

    apply_filters = False
    with st.sidebar:
        with st.expander("🔁 Refresh", expanded=True):
            st.write("Auto-refresh keeps your positions/filters.")
            if st.button("Refresh now"):
                st.session_state.next_refresh_ts = time.time()  # trigger immediately
                st.rerun()

        with st.expander("🧾 Position Tracking", expanded=True):
            st.write("Paste pool **Address** values (one per line).")
            positions_text = st.text_area(
                "My positions (pool addresses):",
                value="\n".join(st.session_state.positions.keys()),
                height=140,
            )
            col_a, col_b = st.columns(2)
            with col_a:
                sync_positions = st.button("Sync list")
            with col_b:
                clear_positions = st.button("Clear all")

            if clear_positions:
                st.session_state.positions = {}
                st.rerun()

            if sync_positions:
                new_addrs = set(parse_positions_text(positions_text))
                old = st.session_state.positions

                updated: Dict[str, Dict[str, Any]] = {}
                for addr in new_addrs:
                    if addr in old:
                        updated[addr] = old[addr]
                    else:
                        updated[addr] = {"entry_ratio_min30": 0.0, "entered_at": ""}

                st.session_state.positions = updated
                st.rerun()

        with st.expander("🔍 Filter Pools", expanded=True):
            st.markdown("**Minimum Pool Age (hours)**")
            st.caption(f"[{min_age} – {max_age}]")
            min_age_input = st.number_input(
                label="",
                min_value=min_age,
                max_value=max_age,
                value=safe_session_number("filter_input_min_age", min_age, max_age, min_age_default),
                step=1,
                key="filter_input_min_age",
            )

            st.markdown("**Minimum Market Cap (MCAP)**")
            st.caption(f"[{min_mcap:,} – {max_mcap:,}]")
            min_mcap_input = st.number_input(
                label="",
                min_value=min_mcap,
                max_value=max_mcap,
                value=safe_session_number("filter_input_min_mcap", min_mcap, max_mcap, min_mcap_default),
                step=1000,
                key="filter_input_min_mcap",
            )

            st.markdown("**Minimum 30 min Volume**")
            st.caption(f"[{min_vol_30:,.2f} – {max_vol_30:,.2f}]")
            min_vol_30_input = st.number_input(
                label="",
                min_value=min_vol_30,
                max_value=max_vol_30,
                value=safe_session_number("filter_input_min_vol_30", min_vol_30, max_vol_30, min_vol_30_default),
                step=1.0,
                key="filter_input_min_vol_30",
            )

            st.markdown("**Minimum Ratio min30**")
            st.caption(f"[{min_ratio_min30:.2f} – {max_ratio_min30:.2f}]")
            min_ratio_min30_input = st.number_input(
                label="",
                min_value=min_ratio_min30,
                max_value=max_ratio_min30,
                value=safe_session_number(
                    "filter_input_min_ratio_min30",
                    min_ratio_min30,
                    max_ratio_min30,
                    min_ratio_min30_default,
                ),
                step=0.01,
                key="filter_input_min_ratio_min30",
            )

            apply_filters = st.button("Apply Filters")

    if apply_filters or "filtered_pairs" not in st.session_state:
        st.session_state.filter_settings = {
            "min_age": min_age_input,
            "min_mcap": min_mcap_input,
            "min_vol_30": min_vol_30_input,
            "min_ratio_min30": min_ratio_min30_input,
        }
        st.session_state.filtered_pairs = [
            p
            for p in pairs
            if parse_created_to_hours(p["Created"]) >= st.session_state.filter_settings["min_age"]
            and p["MCAP"] >= st.session_state.filter_settings["min_mcap"]
            and p["Vol min30"] >= st.session_state.filter_settings["min_vol_30"]
            and p["Ratio min30"] >= st.session_state.filter_settings["min_ratio_min30"]
        ]

    filtered_pairs = st.session_state.filtered_pairs
    if not filtered_pairs:
        st.warning("No pools match your filter settings. Try lowering your filter thresholds.")
        return

    signaled_pairs = compute_signals(filtered_pairs, st.session_state.positions)

    st.markdown("#### Quick Position Actions")
    addresses_in_table = sorted({p.get("Address", "") for p in signaled_pairs if p.get("Address")})
    pick = st.selectbox("Select pool Address:", options=addresses_in_table, index=0 if addresses_in_table else None)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Mark as ENTERED (store entry ratio)"):
            row = next((p for p in signaled_pairs if p.get("Address") == pick), None)
            if row:
                st.session_state.positions[pick] = {
                    "entry_ratio_min30": float(row.get("Ratio min30", 0) or 0),
                    "entered_at": now_utc_iso(),
                }
                st.rerun()

    with col2:
        if st.button("Remove position"):
            if pick in st.session_state.positions:
                st.session_state.positions.pop(pick, None)
                st.rerun()

    sort_options = [
        "Entry Score",
        "Exit Score",
        "Signal",
        "Vol min30",
        "Fee min30",
        "Ratio min30",
        "Liquidity ($)",
        "MCAP",
        "Custom Sort",
        "vol/liq 5m",
        "vol/liq 1h",
        "vol/liq 6h",
        "vol/liq 24h",
    ]

    st.markdown("#### Sort By")
    sort_field = st.radio("Choose sort field:", sort_options, index=0, horizontal=True)
    order = st.radio("Sort order:", options=["Descending", "Ascending"], index=0, horizontal=True)
    reverse = True if order == "Descending" else False

    display_table(signaled_pairs, sort_field, reverse)

    # Finally: perform the ticking auto-refresh (keeps state)
    autorefresh_tick()


if __name__ == "__main__":
    main()
