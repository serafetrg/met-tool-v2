import logging
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

# Only the correct timeframes from Jupiter Ultra API
ULTRA_TIMEFRAMES = ["stats5m", "stats1h", "stats6h", "stats24h"]


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
            hours = int(parts[0])
            return hours

        return 0
    except Exception:
        return 0


# ----------------------------
# Pair processing helpers
# ----------------------------
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
        meteora_link = (
            f"<a href='https://app.meteora.ag/dlmm/{address}' target='_blank' style='white-space:nowrap;'>meteora</a>"
        )
        dexscreener_link = (
            f"<a href='https://dexscreener.com/solana/{mint_x}' target='_blank' style='white-space:nowrap;'>dex</a>"
        )
        jupiter_link = (
            f"<a href='https://jup.ag/swap?sell=EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v&buy={mint_x}' "
            f"target='_blank' style='white-space:nowrap;'>jupiter</a>"
        )
        combined_links = (
            "<div style='text-align:center;line-height:1.2;'>"
            f"{meteora_link}<br>{dexscreener_link}<br>{jupiter_link}</div>"
        )

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

        ultra = ultra_stats.get(mint_x, {})
        ultra_liquidity = float(ultra.get("ultra_liquidity", 0))

        # Calculate vol/liq for the correct timeframes only
        vol_liq_dict: Dict[str, float] = {}
        for tf in ULTRA_TIMEFRAMES:
            tf_stats = ultra.get(tf, {})
            buy_vol = float(tf_stats.get("buyVolume", 0))
            sell_vol = float(tf_stats.get("sellVolume", 0))
            total_vol = buy_vol + sell_vol
            vol_liq = (total_vol / ultra_liquidity) if ultra_liquidity > 0 else 0
            tf_name = tf.replace("stats", "vol/liq ")
            vol_liq_dict[tf_name] = vol_liq

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
            "Vol 1h": float(p.get("volume", {}).get("hour_1", 0)),
            "Fee 1h": float(p.get("fees", {}).get("hour_1", 0)),
            "Ratio 1h": float(p.get("fee_tvl_ratio", {}).get("hour_1", 0)),
            "Custom Sort": None,  # placeholder to keep consistent structure if needed
            "Address": address,
        }
        item.update(vol_liq_dict)
        processed.append(item)

    return processed


# ----------------------------
# Filter helpers
# ----------------------------
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


# ----------------------------
# Table formatting + rendering
# ----------------------------
def format_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatters = {
        "Liquidity ($)": lambda x: f"{x:,.2f}",
        "Vol min30": lambda x: f"{x:,.2f}",
        "Fee min30": lambda x: f"{x:,.2f}",
        "Ratio min30": lambda x: f"{x:.2f}",
        "Base Fee %": lambda x: f"{x:.2f}%",
        "Max Fee %": lambda x: f"{x:.2f}%",
        "MCAP": lambda x: f"{x:,.2f}",
        "Organic Score": lambda x: f"{x:.2f}",
        "LP Ratio": lambda x: f"{x:.0f}% / {100 - float(x):.0f}%" if isinstance(x, (float, int)) else x,
    }

    # Format all vol/liq columns
    for tf in ULTRA_TIMEFRAMES:
        colname = tf.replace("stats", "vol/liq ")
        formatters[colname] = lambda x: f"{float(x):.5f}"

    # Hourly columns to keep
    hourly_number_cols = [
        "Vol 1h",
        "Fee 1h",
        "Ratio 1h",
    ]
    for col in hourly_number_cols:
        formatters[col] = (lambda c: (lambda x: f"{x:,.2f}" if "Vol" in c or "Fee" in c else f"{x:.2f}"))(col)

    for col, func in formatters.items():
        if col in df.columns:
            df[col] = df[col].apply(func)

    return df


def display_table(pairs: List[dict], sort_field: str, reverse: bool) -> None:
    df = pd.DataFrame(pairs)
    df = format_columns(df)

    columns = [
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
        "Vol 1h",
        "Fee 1h",
        "Ratio 1h",
        # Only the correct vol/liq timeframes
        *[tf.replace("stats", "vol/liq ") for tf in ULTRA_TIMEFRAMES],
        "Address",
    ]

    for col in columns:
        if col not in df.columns:
            df[col] = 0.0 if "vol/liq" in col else ""

    df = df[[col for col in columns if col in df.columns]]

    try:
        sort_vals = df[sort_field].str.replace(",", "").str.replace("%", "").astype(float)
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
        .high-ratio-row {
            background-color: #7d3c98 !important;
            color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def row_style(row: pd.Series) -> List[str]:
        try:
            if float(row["Ratio min30"]) > 10:
                return ["background-color: #7d3c98; color: #fff;"] * len(row)
        except Exception:
            pass
        return [""] * len(row)

    styled_df = df.style.apply(row_style, axis=1)

    st.write("### Meteora Pools Table")
    st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.download_button(
        label="Download table as CSV",
        data=df.to_csv(index=False),
        file_name="meteora_pools.csv",
        mime="text/csv",
    )


# ----------------------------
# App
# ----------------------------
def main() -> None:
    st.title("Meteora Pool Scoring Dashboard")
    st.write("This dashboard fetches and scores pools from Meteora, with interactive sorting and filters.")
    st.info("Fetching and processing data. This may take up to 20 seconds on first load...")

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
                value=safe_session_number(
                    "filter_input_min_mcap", min_mcap, max_mcap, min_mcap_default
                ),
                step=1000,
                key="filter_input_min_mcap",
            )

            st.markdown("**Minimum 30 min Volume**")
            st.caption(f"[{min_vol_30:,.2f} – {max_vol_30:,.2f}]")
            min_vol_30_input = st.number_input(
                label="",
                min_value=min_vol_30,
                max_value=max_vol_30,
                value=safe_session_number(
                    "filter_input_min_vol_30", min_vol_30, max_vol_30, min_vol_30_default
                ),
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
        st.warning(
            "No pools match your filter settings. Try lowering the minimum MCAP, pool age, 30 min volume, or min ratio."
        )
        return

    # Only correct sort options
    sort_options = [
        "Vol min30",
        "Fee min30",
        "Ratio min30",
        "Vol 1h",
        "Fee 1h",
        "Ratio 1h",
        "vol/liq 5m",
        "vol/liq 1h",
        "vol/liq 6h",
        "vol/liq 24h",
    ]

    st.markdown("#### Sort By")
    sort_field = st.radio(
        "Choose sort field:",
        sort_options,
        index=0,
        horizontal=True,
    )
    order = st.radio("Sort order:", options=["Descending", "Ascending"], index=0, horizontal=True)
    reverse = True if order == "Descending" else False

    display_table(filtered_pairs, sort_field, reverse)


if __name__ == "__main__":
    main()
