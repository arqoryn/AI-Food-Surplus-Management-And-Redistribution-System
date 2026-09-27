import math
from pathlib import Path

import altair as alt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_preprocessing import (
    DAILY_HISTORY_COLUMNS,
    DAILY_OPERATIONS_COLUMNS,
    ITEM_HISTORY_COLUMNS,
    PRECIPITATION_OPTIONS,
    RECIPIENT_COLUMNS,
    RECORD_COLUMNS,
    WEATHER_OPTIONS,
    append_daily_history,
    append_inventory,
    append_item_history,
    load_csv,
    load_redistribution_records,
    save_redistribution_records,
    validate_columns,
)
from src.daily_demand_model import (
    allocate_daily_to_items,
    calculate_item_shares,
    predict_daily_demand,
    train_or_load_daily_model,
)
from src.inventory import (
    calculate_status,
    create_inventory_batches,
    decrement_remaining,
    load_inventory,
    load_shelf_life,
    update_active_surplus_shelf_life,
)
from src.maps import build_route_map
from src.kitchen_config import (
    BASELINE_SAFETY_BUFFER_PCT,
    BASELINE_WINDOW_DAYS,
    INGREDIENT_UNITS,
    MENU_ITEMS,
    PROTOTYPE_NOTE,
    RAW_INGREDIENTS,
    SUPPORTED_UNITS,
    UNIT_TO_CANONICAL,
    SettingsError,
    SURPLUS_SHELF_LIFE_KEY,
    default_settings,
    ingredient_unit,
    load_settings,
    menu_items_using,
    recipe_for,
    restore_defaults,
    save_settings,
)
from src.raw_inventory import (
    RawInventoryError,
    add_raw_batch,
    available_quantity,
    baseline_demand_by_item,
    baseline_readiness,
    batch_label,
    batch_sequence,
    capacity_summary,
    deduct_for_production,
    expiring_soon,
    load_raw_inventory,
    load_raw_usage,
    normalize_ingredient,
    record_unmapped_batch,
    remove_raw_batch,
    serving_capacity,
    stock_table,
    update_raw_batch,
    usage_by_ingredient,
    usage_by_menu_item,
    usage_over_time,
    usage_reference,
    usage_totals_by_unit,
    usable_usage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "daily_demand_model.joblib"
RAW_INVENTORY_PATH = DATA_DIR / "raw_inventory.csv"
RAW_USAGE_PATH = DATA_DIR / "raw_usage.csv"
SETTINGS_PATH = DATA_DIR / "kitchen_settings.json"


@st.cache_data
def load_operational_data(data_revision):
    history = load_csv(DATA_DIR / "daily_history.csv", DAILY_HISTORY_COLUMNS)
    daily = load_csv(DATA_DIR / "daily_operations.csv", DAILY_OPERATIONS_COLUMNS)
    recipients = load_csv(DATA_DIR / "recipients.csv", RECIPIENT_COLUMNS)
    records = load_redistribution_records(DATA_DIR / "redistribution_records.csv")
    items = load_csv(DATA_DIR / "item_history.csv", ITEM_HISTORY_COLUMNS)
    return history, daily, recipients, records, items


@st.cache_data
def load_surplus_inventory(data_revision):
    inventory = load_inventory(DATA_DIR / "inventory.csv")
    shelf_life = load_shelf_life(DATA_DIR / "surplus_shelf_life.csv")
    return inventory, shelf_life


@st.cache_data
def load_raw_stock(data_revision):
    """Raw ingredients only. Failures degrade to an empty frame, never an error."""
    try:
        return load_raw_inventory(RAW_INVENTORY_PATH)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def operational_data_revision():
    paths = [
        DATA_DIR / "daily_history.csv",
        DATA_DIR / "daily_operations.csv",
        DATA_DIR / "recipients.csv",
        DATA_DIR / "redistribution_records.csv",
        DATA_DIR / "item_history.csv",
        DATA_DIR / "inventory.csv",
        DATA_DIR / "surplus_shelf_life.csv",
        DATA_DIR / "raw_inventory.csv",
        DATA_DIR / "raw_usage.csv",
    ]
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else (str(path), None, None)
        for path in paths
    )


@st.cache_resource
def get_model(history):
    return train_or_load_daily_model(history, MODEL_PATH)



def build_redistribution_plan(surplus, recipients):
    urgency_weight = {"High": 2, "Medium": 1, "Low": 0}
    active = recipients[recipients["active"] == 1].copy()
    ranked = active.assign(
        priority_score=active["priority"].map(urgency_weight).fillna(0) * 100
        + active["current_need"]
        - active["distance_km"]
    ).sort_values(["priority_score", "distance_km"], ascending=[False, True])

    remaining = int(surplus)
    plan = []
    for recipient in ranked.to_dict("records"):
        allocated = min(remaining, int(recipient["current_need"]), int(recipient["capacity"]))
        if allocated <= 0:
            continue
        plan.append(
            {
                "recipient_id": recipient["recipient_id"],
                "Recipient": recipient["name"],
                "Meals allocated": allocated,
                "Need": int(recipient["current_need"]),
                "Priority": recipient["priority"],
                "Distance (km)": float(recipient["distance_km"]),
                "latitude": float(recipient["latitude"]),
                "longitude": float(recipient["longitude"]),
            }
        )
        remaining -= allocated
        if remaining == 0:
            break
    return pd.DataFrame(plan), remaining


def build_fair_round_robin(surplus, recipients):
    """Fair variant of smart matching: cycle priority-ranked recipients in
    rounds so one large recipient cannot swallow the whole batch."""
    urgency_weight = {"High": 2, "Medium": 1, "Low": 0}
    active = recipients[recipients["active"] == 1].copy()
    ranked = active.assign(
        priority_score=active["priority"].map(urgency_weight).fillna(0) * 100
        + active["current_need"]
        - active["distance_km"]
    ).sort_values(["priority_score", "distance_km"], ascending=[False, True])
    rows = ranked.to_dict("records")
    remaining = int(surplus)
    given = {}
    while remaining > 0:
        progressed = False
        needy = [
            r
            for r in rows
            if int(r["current_need"]) - given.get(r["recipient_id"], 0) > 0
            and int(r["capacity"]) - given.get(r["recipient_id"], 0) > 0
        ]
        if not needy:
            break
        chunk = max(1, math.ceil(remaining / len(needy)))
        for recipient in needy:
            if remaining <= 0:
                break
            rid = recipient["recipient_id"]
            give = min(
                remaining,
                int(recipient["current_need"]) - given.get(rid, 0),
                int(recipient["capacity"]) - given.get(rid, 0),
                chunk,
            )
            if give <= 0:
                continue
            given[rid] = given.get(rid, 0) + give
            remaining -= give
            progressed = True
        if not progressed:
            break
    lookup = {r["recipient_id"]: r for r in rows}
    plan = []
    for rid, qty in given.items():
        r = lookup[rid]
        plan.append(
            {
                "recipient_id": rid,
                "Recipient": r["name"],
                "Meals allocated": int(qty),
                "Need": int(r["current_need"]),
                "Priority": r["priority"],
                "Distance (km)": float(r["distance_km"]),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
            }
        )
    plan = pd.DataFrame(plan)
    if not plan.empty:
        plan = plan.sort_values("Meals allocated", ascending=False, kind="stable").reset_index(drop=True)
    return plan, remaining


def split_plan_by_batch(plan, target_batches):
    """Expand a recipient-level plan into recipient x menu_item rows.

    Recipient-level strategies (Even Distribution, Manual Allocation) only
    decide TOTAL meals per recipient. When the selected surplus contains
    several item batches, those totals are attached to the batches in order
    so inventory and saved records stay item-aware.
    """
    columns = [
        "recipient_id",
        "Recipient",
        "Meals allocated",
        "Need",
        "Priority",
        "Distance (km)",
        "latitude",
        "longitude",
        "menu_item",
    ]
    if plan is None or plan.empty or not target_batches:
        return pd.DataFrame(columns=columns)
    available = {}
    for batch in target_batches:
        name = str(batch["menu_item"])
        available[name] = available.get(name, 0) + int(batch["remaining_quantity"])
    rows = []
    for entry in plan.to_dict("records"):
        left = int(entry["Meals allocated"])
        for name, quantity in available.items():
            if left <= 0:
                break
            take = min(left, int(quantity))
            if take <= 0:
                continue
            available[name] = int(quantity) - take
            left -= take
            record = dict(entry)
            record["menu_item"] = name
            record["Meals allocated"] = take
            rows.append(record)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def build_even_plan(surplus, recipients):
    """Unit round-robin split across active recipients, respecting need/capacity."""
    active = recipients[recipients["active"] == 1].copy().sort_values("distance_km")
    rows = active.to_dict("records")
    remaining = int(surplus)
    given = {}
    while remaining > 0:
        progressed = False
        for recipient in rows:
            if remaining <= 0:
                break
            rid = recipient["recipient_id"]
            if int(recipient["current_need"]) - given.get(rid, 0) <= 0:
                continue
            if int(recipient["capacity"]) - given.get(rid, 0) <= 0:
                continue
            given[rid] = given.get(rid, 0) + 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    lookup = {r["recipient_id"]: r for r in rows}
    plan = []
    for rid, qty in given.items():
        r = lookup[rid]
        plan.append(
            {
                "recipient_id": rid,
                "Recipient": r["name"],
                "Meals allocated": int(qty),
                "Need": int(r["current_need"]),
                "Priority": r["priority"],
                "Distance (km)": float(r["distance_km"]),
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
            }
        )
    plan = pd.DataFrame(plan)
    if not plan.empty:
        plan = plan.sort_values("Meals allocated", ascending=False, kind="stable").reset_index(drop=True)
    return plan, remaining


def days_left_text(expiry_date, today):
    try:
        delta = (pd.Timestamp(expiry_date).date() - pd.Timestamp(today).date()).days
    except (TypeError, ValueError):
        return ""
    if delta < 0:
        return f"Expired {-delta} days ago"
    if delta == 0:
        return "0 days"
    return f"{delta} days"


def append_plan_records(plan, operation_date, menu_item=""):
    records = pd.DataFrame(
        {
            "date": operation_date,
            "recipient_id": plan["recipient_id"],
            "allocated_quantity": plan["Meals allocated"],
            "distance_km": plan["Distance (km)"],
            "status": "Planned",
            "menu_item": str(menu_item or ""),
        }
    )
    save_redistribution_records(DATA_DIR / "redistribution_records.csv", records)


def scenario_context(operation_date, weather, precipitation, holiday, special_event):
    day_name = pd.Timestamp(operation_date).day_name()
    holiday_text = "Holiday" if holiday else "No holiday"
    event_text = "Special event" if special_event else "No special event"
    return f"{weather} {day_name} · {precipitation} · {holiday_text} · {event_text}"


def end_of_day_demand_reading(prediction_deviation):
    """Plain-language reading of actual consumption versus predicted demand."""
    if prediction_deviation == 0:
        return "Actual consumption matched the predicted demand."
    direction = "higher" if prediction_deviation > 0 else "lower"
    return f"Actual consumption was {abs(prediction_deviation):,} meals {direction} than predicted demand."


def end_of_day_surplus_reading(actual_surplus):
    """Plain-language reading of the meals that were produced but not consumed."""
    if actual_surplus == 0:
        return "Every meal produced was consumed, so no surplus remained."
    return (
        f"{actual_surplus:,} meals were produced but not consumed, leaving "
        f"{actual_surplus:,} meals of surplus."
    )


def end_of_day_production_reading(production_deviation):
    """Plain-language reading of actual production versus the recommendation."""
    if production_deviation == 0:
        return "Actual production matched the recommended production level."
    direction = "above" if production_deviation > 0 else "below"
    return (
        f"Actual production was {abs(production_deviation):,} meals {direction} "
        "the recommended production level."
    )


def end_of_day_variance_status(deviation, subject):
    """Short status label, for example 'Matched prediction' or 'Below recommendation'."""
    if deviation == 0:
        return f"Matched {subject}"
    return f"{'Above' if deviation > 0 else 'Below'} {subject}"


def end_of_day_variance_note(deviation, subject):
    """Formula-side note, for example 'Above recommendation by 30 meals'."""
    if deviation == 0:
        return f"Matched {subject}"
    return f"{'Above' if deviation > 0 else 'Below'} {subject} by {abs(deviation):,} meals"


def navigate_to(section):
    st.session_state["active_section"] = section


def apply_inventory_decrement(path, inventory_id, quantity):
    """Local persist wrapper built on existing `decrement_remaining` helper."""
    inventory = load_inventory(path)
    matches = inventory[inventory["inventory_id"].astype(str) == str(inventory_id)]
    if matches.empty:
        raise ValueError(f"Inventory batch {inventory_id} was not found.")
    row_index = matches.index[0]
    new_remaining = decrement_remaining(int(inventory.at[row_index, "remaining_quantity"]), quantity)
    inventory.at[row_index, "remaining_quantity"] = new_remaining
    inventory.at[row_index, "status"] = calculate_status(
        new_remaining, inventory.at[row_index, "expiry_date"], pd.Timestamp.today().date()
    )
    inventory.to_csv(path, index=False)
    return new_remaining


def start_redistribute_all(batches):
    """Prepare every allocatable batch for item-separated review."""
    st.session_state.pop("selected_inventory_item", None)
    st.session_state["redistribute_all_items"] = [
        {
            "inventory_id": b["_inventory_id"],
            "menu_item": b["Menu Item"],
            "remaining_quantity": int(b["Remaining Quantity"]),
        }
        for b in batches
    ]
    st.session_state["active_section"] = "Recipient Matching"


def start_inventory_redistribution(inventory_id, menu_item, remaining_quantity):
    """Store the selected surplus batch and navigate to Recipient Matching.

    Item-aware redistribution itself is a future phase; this only prepares
    the selected inventory context in session state.
    """
    st.session_state.pop("redistribute_all_items", None)
    st.session_state["selected_inventory_item"] = {
        "inventory_id": inventory_id,
        "menu_item": menu_item,
        "remaining_quantity": int(remaining_quantity),
    }
    st.session_state["active_section"] = "Recipient Matching"


def metric_card(label, value, detail, accent):
    tint = {
        "#4d9fff": "rgba(77, 159, 255, .12)",
        "#50c878": "rgba(80, 200, 120, .12)",
        "#e6ad45": "rgba(230, 173, 69, .12)",
        "#ef6f6c": "rgba(239, 111, 108, .12)",
    }.get(accent, "rgba(255, 255, 255, .06)")
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent: {accent}; --tint: {tint};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def demand_trend_chart(history, model, operation_date, predicted_demand, window=14):
    required_columns = [
        "date", "predicted_demand", "actual_production",
    ]
    if history.empty or any(column not in history.columns for column in required_columns):
        return None

    historical = history.copy()
    historical["date"] = pd.to_datetime(historical["date"], errors="coerce")
    for column in ["predicted_demand", "actual_production"]:
        historical[column] = pd.to_numeric(historical[column], errors="coerce")
    historical = historical.replace([float("inf"), float("-inf")], pd.NA)
    historical = historical.dropna(subset=required_columns).sort_values("date").tail(window)
    if historical.empty:
        return None

    predicted_rows = [
        {
            "date": historical_row["date"],
            "series": "Predicted demand",
            "meals": float(historical_row["predicted_demand"]),
        }
        for _, historical_row in historical.iterrows()
    ]

    forecast_date = pd.Timestamp(operation_date)
    predicted_rows = [row for row in predicted_rows if row["date"] != forecast_date]
    predicted_rows.append(
        {
            "date": forecast_date,
            "series": "Predicted demand",
            "meals": int(predicted_demand),
        }
    )
    production_rows = [
        {
            "date": historical_row["date"],
            "series": "Actual production",
            "meals": float(historical_row["actual_production"]),
        }
        for _, historical_row in historical.iterrows()
    ]
    chart_data = pd.DataFrame(production_rows + predicted_rows).dropna(subset=["date", "meals"])
    if chart_data.empty:
        return None

    color_scale = alt.Scale(
        domain=["Actual production", "Predicted demand"],
        range=["#50c878", "#4d9fff"],
    )
    base = alt.Chart(chart_data).encode(
        x=alt.X(
            "date:T",
            title="Date",
            axis=alt.Axis(format="%d %b", labelColor="#9eacbf", titleColor="#9eacbf"),
        ),
        y=alt.Y(
            "meals:Q",
            title="Meals",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(labelColor="#9eacbf", titleColor="#9eacbf"),
        ),
        color=alt.Color(
            "series:N",
            scale=color_scale,
            legend=alt.Legend(title=None, labelColor="#d7e0eb"),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("meals:Q", title="Meals", format=",.0f"),
        ],
    )
    production_line = base.transform_filter(
        alt.datum.series == "Actual production"
    ).mark_line(point=True, strokeWidth=2.5)
    prediction_line = base.transform_filter(
        alt.datum.series == "Predicted demand"
    ).mark_line(point=True, strokeDash=[2, 4], strokeWidth=2.5)
    return alt.layer(production_line, prediction_line).properties(height=360).interactive()


def planning_review_rows(history, model, start_date, end_date, current_date, current_values):
    required_columns = [
        "date", "predicted_demand", "actual_production", "actual_consumption",
        "surplus_quantity", "holiday", "special_event",
        "weather", "precipitation",
    ]
    if history.empty or any(column not in history.columns for column in required_columns):
        historical = pd.DataFrame(columns=required_columns)
    else:
        historical = history.copy()
        historical["date"] = pd.to_datetime(historical["date"], errors="coerce")
        for column in [
            "predicted_demand", "actual_production", "actual_consumption", "surplus_quantity",
            "holiday", "special_event",
        ]:
            historical[column] = pd.to_numeric(historical[column], errors="coerce")
        historical = historical.replace([float("inf"), float("-inf")], pd.NA)
        historical = historical.dropna(subset=required_columns)
        historical = historical[
            (historical["date"] >= pd.Timestamp(start_date))
            & (historical["date"] <= pd.Timestamp(end_date))
        ].sort_values("date")

    rows = []
    for _, historical_row in historical.iterrows():
        rows.append(
            {
                "date": historical_row["date"],
                "predicted_demand": int(historical_row["predicted_demand"]),
                "actual_production": float(historical_row["actual_production"]),
                "planned_production": None,
                "actual_consumption": float(historical_row["actual_consumption"]),
                "surplus_quantity": float(historical_row["surplus_quantity"]),
                "holiday": bool(historical_row["holiday"]),
                "special_event": bool(historical_row["special_event"]),
                "weather": historical_row["weather"],
                "precipitation": historical_row["precipitation"],
                "source": "Historical record",
            }
        )

    current_timestamp = pd.Timestamp(current_date)
    if pd.Timestamp(start_date) <= current_timestamp <= pd.Timestamp(end_date):
        has_current_record = any(row["date"] == current_timestamp for row in rows)
        if not has_current_record:
            rows.append(
                {
                    "date": current_timestamp,
                    "predicted_demand": int(current_values["predicted_demand"]),
                    "actual_production": None,
                    "planned_production": float(current_values["planned_production"]),
                    "actual_consumption": None,
                    "surplus_quantity": None,
                    "holiday": bool(current_values["holiday"]),
                    "special_event": bool(current_values["special_event"]),
                    "weather": current_values["weather"],
                    "precipitation": current_values["precipitation"],
                    "source": "Current scenario",
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)



def planning_review_chart(review_rows):
    chart_rows = []
    for _, review_row in review_rows.iterrows():
        chart_rows.extend(
            [
                {
                    "date": review_row["date"],
                    "series": "Actual production",
                    "meals": review_row["actual_production"],
                },
                {
                    "date": review_row["date"],
                    "series": "Predicted demand",
                    "meals": review_row["predicted_demand"],
                },
            ]
        )
    chart_data = pd.DataFrame(chart_rows).dropna(subset=["date", "meals"])
    if chart_data.empty:
        return None
    chart_data["date_label"] = chart_data["date"].dt.strftime("%d %b")
    date_order = (
        chart_data[["date", "date_label"]]
        .drop_duplicates()
        .sort_values("date")["date_label"]
        .tolist()
    )
    series_order = ["Predicted demand", "Actual production"]

    base = alt.Chart(chart_data).encode(
        x=alt.X("date_label:N", title="Date", sort=date_order, scale=alt.Scale(paddingInner=0.35, paddingOuter=0.15)),
        y=alt.Y("meals:Q", title="Meals", scale=alt.Scale(zero=True), stack=None),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(
                domain=series_order,
                range=["#4d9fff", "#50c878"],
            ),
            legend=alt.Legend(title=None),
        ),
        xOffset=alt.XOffset("series:N", sort=series_order),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("meals:Q", title="Meals", format=",.0f"),
        ],
    )
    return base.mark_bar(size=18).resolve_scale(y="shared").properties(height=360).interactive()


def redistribution_trend_figure(delivered_records):
    if delivered_records.empty:
        return None

    trend = (
        delivered_records.assign(date=pd.to_datetime(delivered_records["date"]))
        .groupby("date", as_index=False)["allocated_quantity"]
        .sum()
        .sort_values("date")
    )
    figure, axis = plt.subplots(figsize=(10, 3.4))
    figure.patch.set_facecolor("#111b2a")
    axis.set_facecolor("#111b2a")
    axis.plot(
        trend["date"],
        trend["allocated_quantity"],
        color="#e6ad45",
        linewidth=2.4,
        marker="o",
        markersize=6,
    )
    value_range = max(trend["allocated_quantity"].max() - trend["allocated_quantity"].min(), 1)
    padding = max(value_range * 0.2, 10)
    axis.set_ylim(max(0, trend["allocated_quantity"].min() - padding), trend["allocated_quantity"].max() + padding)
    axis.set_ylabel("Meals delivered")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axis.grid(axis="y", color="#2a3a50", linewidth=0.8, alpha=0.7)
    axis.grid(axis="x", visible=False)
    axis.tick_params(colors="#9eacbf", labelsize=9)
    axis.yaxis.label.set_color("#9eacbf")
    for spine in axis.spines.values():
        spine.set_color("#2a3a50")
    figure.autofmt_xdate(rotation=0, ha="center")
    figure.tight_layout()
    return figure


def impact_period_cutoff(label, end_date):
    days = {"7 Days": 7, "30 Days": 30, "90 Days": 90}.get(label, None)
    if days is None:
        return None
    return (pd.Timestamp(end_date) - pd.Timedelta(days=days - 1)).date()


def impact_filter_by_period(frame, date_col, start_date):
    if frame.empty or start_date is None:
        return frame
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    return frame[dates.dt.date >= start_date].copy()


def build_production_breakdown(selected_menu_items, item_history, daily_prediction, daily_recommendation):
    """Item-level breakdown allocated from the daily totals.

    Historical consumption shares of the selected items are renormalized to 1, then
    the daily predicted demand and the daily recommended production are allocated
    with deterministic integer reconciliation. Returns None when nothing is selected.
    """
    if not selected_menu_items or item_history.empty:
        return None

    all_shares = calculate_item_shares(item_history)
    ordered_items = []
    for item in selected_menu_items:
        if item in all_shares and item not in ordered_items:
            ordered_items.append(item)
    if not ordered_items:
        return None

    total_share = sum(all_shares[item] for item in ordered_items)
    if total_share <= 0:
        return None
    selected_shares = {item: all_shares[item] / total_share for item in ordered_items}

    predicted_allocation = allocate_daily_to_items(int(daily_prediction), selected_shares)
    recommended_allocation = allocate_daily_to_items(int(daily_recommendation), selected_shares)

    breakdown = pd.DataFrame(
        [
            {
                "Menu Item": item,
                "Predicted": predicted_allocation[item],
                "Recommended": recommended_allocation[item],
            }
            for item in ordered_items
        ],
        columns=["Menu Item", "Predicted", "Recommended"],
    )
    totals = pd.DataFrame(
        [
            {
                "Menu Item": "TOTAL",
                "Predicted": int(breakdown["Predicted"].sum()),
                "Recommended": int(breakdown["Recommended"].sum()),
            }
        ],
        columns=["Menu Item", "Predicted", "Recommended"],
    )
    return pd.concat([breakdown, totals], ignore_index=True)


def render_production_breakdown(breakdown):
    if breakdown is None:
        st.warning(
            "No menu items are selected for today. Choose today's menu items in the "
            "Kitchen scenario to see the item-level production breakdown."
        )
        return
    st.dataframe(
        breakdown.style.format({"Predicted": "{:,.0f}", "Recommended": "{:,.0f}"}),
        hide_index=True,
        width="stretch",
    )
    st.caption("Allocated from today's total demand prediction using historical item-consumption shares.")


st.set_page_config(page_title="SMAR System (Surplus Management And Redistribution System)", page_icon=":material/compost:", layout="wide")
UI_THEME = "Dark"
LIGHT = False
BG = "#f3f5f9" if LIGHT else "#080d16"
PANEL = "#ffffff" if LIGHT else "#111b2a"
PANEL2 = "#eef1f6" if LIGHT else "#101a29"
BORDER = "#dbe2ec" if LIGHT else "#26364c"
TXT = "#182743" if LIGHT else "#e7edf5"
MUTED = "#5d6f87" if LIGHT else "#9eacbf"
st.markdown(
    f"""
    <style>
    :root {{ --smar-bg: {BG}; --smar-panel: {PANEL}; --smar-panel2: {PANEL2}; --smar-border: {BORDER}; --smar-txt: {TXT}; --smar-muted: {MUTED}; }}
    .stApp {{ background: var(--smar-bg); color: var(--smar-txt); }}
    .stApp, .stApp p, .stApp label, .stApp textarea, .stApp input {{ font-family: "Inter", "Aptos", "Segoe UI", sans-serif; }}
    .block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }}
    h1 {{ font-size: 2rem !important; font-weight: 800 !important; letter-spacing: -0.02em; }}
    h2 {{ font-size: 1.25rem !important; font-weight: 700 !important; margin: 1.8rem 0 .7rem; }}
    h3 {{ font-size: 1.02rem !important; font-weight: 600 !important; margin: 1.3rem 0 .55rem; }}
    h5, h6 {{ letter-spacing: .1em; text-transform: uppercase; font-size: .72rem !important; color: var(--smar-muted) !important; }}
    [data-testid="stMetric"] {{ border-radius: .8rem; padding: 1rem 1.1rem; }}
    [data-testid="stMetricValue"] {{ font-size: 1.55rem !important; font-weight: 800 !important; }}
    div[data-testid="stContainer"][data-border="true"], [data-testid="stExpander"] {{ border-radius: .9rem !important; box-shadow: 0 1px 2px rgba(16,24,40,.06); }}
    div[data-testid="stContainer"][data-border="true"] {{ padding: 1.25rem 1.3rem !important; margin-bottom: 1.1rem; }}
    hr {{ margin: 1.5rem 0 !important; }}
    .stButton > button {{ border-radius: .65rem; font-weight: 600; min-height: 2.6rem; }}
    .stButton > button[kind="primary"] {{ min-height: 3rem; font-size: 1rem; }}
    .stDataFrame, .stTable {{ border-radius: .7rem; overflow: hidden; }}
    [data-testid="stSidebar"] {{ background: var(--smar-panel2); border-right: 1px solid var(--smar-border); width: 300px; }}
    [data-testid="stSidebar"] hr {{ border-color: var(--smar-border); }}
    [data-testid="stSidebar"] h3 {{ color: var(--smar-txt); letter-spacing: .08em; text-transform: uppercase; }}
    .sidebar-brand {{ padding: .35rem .25rem .2rem; }}
    .sidebar-brand-mark {{ color: var(--smar-txt); font-size: 1.7rem; font-weight: 800; letter-spacing: .08em; line-height: 1; }}
    .sidebar-brand-name {{ color: var(--smar-muted); font-size: .62rem; font-weight: 700; letter-spacing: .13em; line-height: 1.55; margin-top: .5rem; text-transform: uppercase; }}
    [data-testid="stSidebar"] [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: .25rem; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{ border: 1px solid transparent; border-radius: .6rem; color: var(--smar-muted); margin: .12rem 0; padding: .6rem .75rem; font-weight: 500; transition: background .15s ease, color .15s ease; }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: var(--smar-panel); color: var(--smar-txt); }}
    [data-testid="stSidebar"] [data-testid="stForm"] {{ border: 1px solid var(--smar-border); border-radius: .65rem; padding: .7rem; background: var(--smar-panel); }}
    .block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }}
    .hero {{ padding: .15rem 0 1.45rem; border-bottom: 1px solid var(--smar-border); margin-bottom: 1.7rem; }}
    .hero h1 {{ margin: 0; color: var(--smar-txt); font-size: 2.25rem; letter-spacing: .01em; line-height: 1.05; }}
    .hero .product-name {{ margin: .45rem 0 0; color: var(--smar-txt); font-size: 1rem; font-weight: 600; letter-spacing: .02em; }}
    .hero .hero-copy {{ margin: .55rem 0 0; color: var(--smar-muted); font-size: .9rem; }}
    .hero-date {{ color: var(--smar-muted); text-align: right; padding-top: .35rem; font-size: .88rem; }}
    .section-label {{ color: #1f7a4d; font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
    .kpi-card {{ min-height: 148px; box-sizing: border-box; padding: 1.1rem 1.2rem; background: var(--smar-panel); border: 1px solid var(--smar-border); border-top: 3px solid var(--accent); border-radius: .7rem; box-shadow: 0 1px 2px rgba(16,24,40,.06); }}
    .kpi-label {{ color: var(--smar-muted); font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }}
    .kpi-value {{ color: var(--smar-txt); font-size: 2rem; font-weight: 700; line-height: 1.2; margin-top: .7rem; }}
    .kpi-detail {{ color: var(--smar-muted); font-size: .82rem; margin-top: .45rem; }}
    .section-heading {{ border-left: 3px solid var(--section-accent); padding-left: .8rem; }}
    .workspace-heading {{ margin-top: .7rem; margin-bottom: 1.3rem; }}
    .workspace-heading h2 {{ margin-bottom: .25rem; color: var(--smar-txt); }}
    .workspace-heading p {{ color: var(--smar-muted); margin: 0; }}
    [data-testid="stMetric"] {{ min-height: 92px; background: var(--smar-panel); border: 1px solid var(--smar-border); border-radius: .65rem; padding: .9rem 1rem; }}
    [data-testid="stMetricLabel"] {{ color: var(--smar-muted); }}
    [data-testid="stMetricValue"] {{ color: var(--smar-txt); }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--smar-panel); border-color: var(--smar-border); border-radius: .7rem; }}
    .decision-title {{ color: #1f7a4d; font-size: .76rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }}
    .decision-value {{ color: var(--smar-txt); font-size: 1.35rem; font-weight: 700; margin: .5rem 0 .25rem; }}
    .decision-note {{ color: var(--smar-muted); font-size: .88rem; }}
    .planning-breakdown {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0 1.35rem; padding: 1rem 1.1rem; background: var(--smar-panel); border: 1px solid var(--smar-border); border-radius: .7rem; }}
    .planning-stat {{ border-left: 2px solid var(--planning-accent); padding-left: .75rem; }}
    .planning-stat-label {{ color: var(--smar-muted); font-size: .75rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }}
    .planning-stat-value {{ color: var(--smar-txt); font-size: 1.55rem; font-weight: 750; line-height: 1.15; margin-top: .35rem; }}
    .status-chip {{ display: inline-block; border: 1px solid currentColor; border-radius: 999px; font-size: .72rem; font-weight: 700; letter-spacing: .06em; padding: .25rem .55rem; text-transform: uppercase; }}
    @media (max-width: 1200px) {{ .eod-breakdown {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 700px) {{ .eod-breakdown {{ grid-template-columns: minmax(0, 1fr); }} }}
    .eod-stat {{ background: var(--smar-panel); border: 1px solid var(--smar-border); border-left: 3px solid var(--eod-accent); border-radius: .65rem; padding: .9rem 1rem; }}
    .eod-stat-label {{ color: var(--smar-muted); font-size: .74rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }}
    .eod-stat-value {{ color: var(--smar-txt); font-size: 1.45rem; font-weight: 700; line-height: 1.15; margin-top: .4rem; }}
    .eod-stat-note {{ color: var(--smar-muted); font-size: .76rem; margin-top: .3rem; }}
    .eod-result-label {{ color: var(--smar-muted); font-size: .76rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
    .eod-result-value {{ color: var(--smar-txt); font-size: 1.9rem; font-weight: 700; line-height: 1.2; margin: .45rem 0 .4rem; }}
    .eod-pair {{ color: var(--smar-muted); font-size: .84rem; margin: .15rem 0 0; }}
    .eod-pair strong {{ color: var(--smar-txt); font-weight: 700; }}
    .eod-block-heading {{ color: var(--smar-muted); font-size: .7rem; font-weight: 700; letter-spacing: .1em; margin: 1.05rem 0 .4rem; text-transform: uppercase; }}
    .eod-formula {{ background: var(--smar-panel2); border: 1px solid var(--smar-border); border-radius: .5rem; color: var(--smar-txt); font-family: "Cascadia Code", "Consolas", monospace; font-size: .84rem; line-height: 1.7; padding: .6rem .8rem; }}
    .eod-formula-note {{ color: var(--smar-muted); display: block; font-family: "Aptos", "Segoe UI", sans-serif; font-size: .78rem; margin-top: .35rem; }}
    .eod-meaning {{ color: var(--smar-txt); font-size: .9rem; line-height: 1.6; }}
    .eod-note {{ color: var(--smar-muted); font-size: .85rem; line-height: 1.6; margin-top: .5rem; }}
    .eod-interpretation {{ color: var(--smar-txt); font-size: .95rem; line-height: 1.7; }}
    .quick-action-label {{ color: var(--smar-muted); font-size: .76rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
    .stButton > button {{ border-radius: .65rem; font-weight: 600; min-height: 2.6rem; }}
    .stButton > button[kind="primary"] {{ min-height: 3rem; font-size: 1rem; }}
    [data-testid="stDataFrame"] {{ border: 1px solid var(--smar-border); border-radius: .7rem; overflow: hidden; }}
    [data-testid="stMetric"] {{ background: var(--smar-panel); border: 1px solid var(--smar-border); }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--smar-panel); border-color: var(--smar-border); }}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    history, daily_operations, recipients, records, item_history = load_operational_data(operational_data_revision())
    inventory, shelf_life = load_surplus_inventory(operational_data_revision())
    raw_stock = load_raw_stock(operational_data_revision())
    model = get_model(history)
except (FileNotFoundError, ValueError) as error:
    st.error(f"Data configuration error: {error}")
    st.stop()

default_operation = daily_operations.iloc[-1]
# The active menu is the canonical four-item menu; history only refines it.
available_menu_items = [item for item in MENU_ITEMS]
@st.dialog("Kitchen Scenario")
def open_kitchen_scenario():
    st.caption("Today's operating inputs")
    weather_default = st.session_state.get("scenario_weather_choice", default_operation["weather"])
    weather = st.selectbox("Weather", WEATHER_OPTIONS, index=WEATHER_OPTIONS.index(weather_default) if weather_default in WEATHER_OPTIONS else 0, key="dlg_weather")
    st.session_state["scenario_weather_choice"] = weather
    precipitation_options = PRECIPITATION_OPTIONS[weather]
    default_precipitation = default_operation["precipitation"]
    previous_precipitation = st.session_state.get("scenario_precipitation_choice")
    if previous_precipitation is None:
        precipitation_index = (
            precipitation_options.index(default_precipitation) if default_precipitation in precipitation_options else 0
        )
    elif previous_precipitation in precipitation_options:
        precipitation_index = precipitation_options.index(previous_precipitation)
    else:
        precipitation_index = 0
    precipitation = st.selectbox(
        "Precipitation",
        precipitation_options,
        index=precipitation_index,
        key=f"scenario_precipitation_{weather}",
    )
    st.session_state["scenario_precipitation_choice"] = precipitation
    with st.form("scenario_form"):
        operation_date = st.date_input("Operation date", value=st.session_state.get("scenario_operation_date", pd.to_datetime(default_operation["date"]).date()))
        planned_production = st.number_input(
            "Planned production (meals)", min_value=0, value=int(st.session_state.get("scenario_production", default_operation["planned_quantity"])), step=10
        )
        holiday = st.checkbox("Holiday", value=bool(st.session_state.get("scenario_holiday", default_operation["holiday"])))
        special_event = st.checkbox("Special event or high-footfall day", value=bool(st.session_state.get("scenario_special_event", default_operation["special_event"])))
        safety_buffer = st.number_input(
            "Safety buffer (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get("scenario_safety_buffer", default_operation.get("safety_buffer", 5.0))),
            step=0.5,
            format="%.1f",
        )
        selected_menu_items_local = st.multiselect(
            "Menu items served today",
            available_menu_items,
            default=st.session_state.get("scenario_menu_items", available_menu_items),
        )
        submitted = st.form_submit_button("Update Forecast", type="primary", icon=":material/refresh:")
        if submitted:
            st.session_state["scenario_operation_date"] = operation_date
            st.session_state["scenario_production"] = int(planned_production)
            st.session_state["scenario_holiday"] = int(holiday)
            st.session_state["scenario_special_event"] = int(special_event)
            st.session_state["scenario_safety_buffer"] = float(safety_buffer)
            st.session_state["scenario_menu_items"] = selected_menu_items_local
            st.rerun()


# Scenario state bridge: the dialog edits scenario_* session keys; the rest of
# the app reads these module-level names (same defaults as before the popup).
operation_date = st.session_state.get("scenario_operation_date", pd.to_datetime(default_operation["date"]).date())
weather = st.session_state.get("scenario_weather_choice", default_operation["weather"])
precipitation = st.session_state.get("scenario_precipitation_choice", default_operation["precipitation"])
planned_production = int(st.session_state.get("scenario_production", default_operation["planned_quantity"]))
holiday = int(st.session_state.get("scenario_holiday", default_operation["holiday"]))
special_event = int(st.session_state.get("scenario_special_event", default_operation["special_event"]))
safety_buffer = float(st.session_state.get("scenario_safety_buffer", default_operation.get("safety_buffer", 5.0)))
selected_menu_items = st.session_state.get("scenario_menu_items", available_menu_items)


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-mark">SMAR</div>
            <div class="sidebar-brand-name">Surplus Management<br>&amp; Redistribution</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    active_section = st.radio(
        "Workspace",
        ["Overview / Today", "Demand & Planning", "Raw Inventory", "Recipient Matching", "Surplus Inventory", "Impact & Analytics", "End of the Day", "Settings"],
        key="active_section",
        label_visibility="collapsed",
    )
    nav_accent = {
        "Overview / Today": "#50c878",
        "Demand & Planning": "#4d9fff",
        "Raw Inventory": "#c08a5e",
        "Recipient Matching": "#ef6f6c",
        "Surplus Inventory": "#e6ad45",
        "Impact & Analytics": "#e6ad45",
        "End of the Day": "#70d49b",
        "Settings": "#9eacbf",
    }[active_section]
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
            background: #182536;
            border-left: 3px solid {nav_accent};
            color: #f4f7fb;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    if active_section == "Overview / Today":
        st.markdown("#### Kitchen scenario")
        if st.button("Kitchen Scenario", width="stretch"):
            open_kitchen_scenario()
    with st.expander("Data sources"):
        st.caption(f"Historical demand · {len(history)} records")
        st.caption(f"Menu items · {len(available_menu_items)} available")
        st.caption(f"Active recipients · {int(recipients['active'].sum())}")
        st.caption(f"Last delivery · {records['date'].max()}")

prediction_input = pd.DataFrame(
    [
        {
            "date": operation_date,
            "holiday": int(holiday),
            "special_event": int(special_event),
            "weather": weather,
            "precipitation": precipitation,
        }
    ]
)
predicted_demand = predict_daily_demand(model, prediction_input)

expected_surplus = max(int(planned_production) - predicted_demand, 0)
waste_risk = min(expected_surplus / max(int(planned_production), 1) * 100, 100)
recommended_production = math.ceil(predicted_demand * (1 + safety_buffer / 100))
avoidable_surplus = max(int(planned_production) - recommended_production, 0)
redistribution_plan, unallocated_surplus = build_redistribution_plan(expected_surplus, recipients)
redistributed_meals = int(redistribution_plan["Meals allocated"].sum()) if not redistribution_plan.empty else 0
co2_avoided = redistributed_meals * 0.5
production_breakdown = build_production_breakdown(
    selected_menu_items, item_history, predicted_demand, recommended_production
)

# --- Effective configuration (Settings overrides centralized defaults) ---
try:
    kitchen_settings = load_settings(SETTINGS_PATH)
except (OSError, ValueError):  # never block the app on a settings problem
    kitchen_settings = default_settings()
EXPIRING_SOON_ACTIVE = int(kitchen_settings["expiring_soon_days"])
APPROACHING_EXPIRY_ACTIVE = int(kitchen_settings["approaching_expiry_days"])
# Settings override the existing surplus_shelf_life.csv per menu item. Where no
# override exists the CSV value stands, so there is still one shelf-life
# mechanism. Overrides affect newly created surplus batches only; existing
# surplus rows keep their original expiry dates.
for _item, _days in (kitchen_settings.get(SURPLUS_SHELF_LIFE_KEY) or {}).items():
    shelf_life[_item] = int(_days)

# --- Raw inventory capacity (operational only; never changes the prediction) ---
raw_today = pd.Timestamp(operation_date).date()
raw_errors: list = []
try:
    raw_capacities = capacity_summary(raw_stock, raw_today)
except (RawInventoryError, KeyError, TypeError, ValueError) as error:
    raw_capacities = []
    raw_errors.append(f"Production capacity could not be calculated: {error}")
raw_baseline = baseline_demand_by_item(
    item_history,
    raw_today,
    int(kitchen_settings["baseline_window_days"]),
    float(kitchen_settings["safety_buffer_pct"]),
)
raw_readiness = baseline_readiness(raw_capacities, raw_baseline)
capacity_by_item = {entry["menu_item"]: int(entry["capacity"]) for entry in raw_capacities}


def predicted_by_menu_item():
    """Today's item-level split, from the existing prediction only."""
    if production_breakdown is None:
        return {}
    frame = production_breakdown[production_breakdown["Menu Item"] != "TOTAL"]
    return {
        str(row["Menu Item"]): int(row["Predicted"])
        for _, row in frame.iterrows()
    }

section_accent = {
    "Overview / Today": "#50c878",
    "Demand & Planning": "#4d9fff",
    "Raw Inventory": "#c08a5e",
    "Recipient Matching": "#ef6f6c",
    "Surplus Inventory": "#e6ad45",
    "Impact & Analytics": "#e6ad45",
    "End of the Day": "#70d49b",
    "Settings": "#9eacbf",
}[active_section]
st.markdown(
    f"""
    <div class="hero">
        <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start;">
            <div>
                <h1>SMAR System</h1>
                <p class="product-name">Surplus Management &amp; Redistribution</p>
                <p class="hero-copy">Smarter food planning, surplus detection and sustainable redistribution.</p>
            </div>
            <div class="hero-date">{pd.Timestamp(operation_date).strftime('%A, %d %b %Y')}</div>
        </div>
    </div>
    <div class="section-label" style="color:{section_accent};">{active_section}</div>
    """,
    unsafe_allow_html=True,
)

prediction_context = scenario_context(operation_date, weather, precipitation, holiday, special_event)
RAW_TONE_COLORS = {
    "expired": ("#1a1f27", "#8a94a6"),
    "expiring": ("#3a1c1b", "#ff9a95"),
    "approaching": ("#3a2f16", "#f2c66a"),
    "healthy": ("#14301f", "#8fe3b4"),
    "depleted": ("#222b38", "#c9d3e0"),
    "unknown": ("#1b2330", "#b8c4d6"),
}


def raw_notification_payload(raw_stock, raw_today, raw_capacities, today_split, capacity_by_item):
    """Build the Notification Center payload. Each notification appears once."""
    payload = {"stock": [], "expiry": [], "prediction": []}
    try:
        view = stock_table(raw_stock, raw_today)
    except (KeyError, TypeError, ValueError):
        view = pd.DataFrame()
    if not view.empty:
        canonical = view["canonical_ingredient"].fillna("").astype(str)
        unmapped = view[canonical.str.len() == 0]
        if not unmapped.empty and kitchen_settings.get("stock_alerts", True):
            names = ", ".join(f"'{n}'" for n in unmapped["ingredient"].astype(str))
            payload["stock"].append(
                f"{len(unmapped)} batch(es) contain an ingredient that is not in the "
                f"canonical list: {names}. The batch is preserved and excluded from "
                "recipe calculations until corrected."
            )
    for notice in expiring_soon(raw_stock, raw_today, EXPIRING_SOON_ACTIVE, APPROACHING_EXPIRY_ACTIVE):
        if not kitchen_settings.get("expiry_alerts", True):
            break
        if notice["tone"] == "expired":
            head = (
                f"{notice['ingredient']} - Expired - {notice['remaining']:g} "
                f"{notice['unit']} remaining - expired {notice['expired_days_ago']} day(s) ago."
            )
        elif notice["tone"] == "expiring":
            head = (
                f"{notice['ingredient']} - expires in {notice['days_left']} day(s) - "
                f"{notice['remaining']:g} {notice['unit']} remaining."
            )
        else:
            head = (
                f"{notice['ingredient']} - approaching expiry - expires in "
                f"{notice['days_left']} day(s) - {notice['remaining']:g} {notice['unit']} remaining."
            )
        options = notice["menu_options"]
        if options:
            head += f" {options[0]} uses this ingredient."
        payload["expiry"].append({"text": head, "tone": notice["tone"]})
    for item, predicted in sorted(today_split.items()):
        if not kitchen_settings.get("prediction_alerts", True):
            break
        capacity = capacity_by_item.get(item, 0)
        if predicted > capacity:
            payload["prediction"].append(
                f"Model predicted {predicted:,} {item}, but current raw inventory supports "
                f"only {capacity:,} servings. Consider adding ingredients."
            )
    return payload


def raw_notification_count(payload):
    return (
        len(payload["stock"]) + len(payload["expiry"]) + len(payload["prediction"])
    )



if active_section == "Overview / Today":
    st.markdown("### Today's situation")
    kpi_columns = st.columns(4)
    with kpi_columns[0]:
        metric_card("Predicted demand", f"{predicted_demand:,} meals", prediction_context, "#4d9fff")
    with kpi_columns[1]:
        metric_card(
            "Recommended production",
            f"{recommended_production:,} meals",
            f"Planned: {int(planned_production):,} · Buffer: {recommended_production - predicted_demand:,}",
            "#50c878",
        )
    with kpi_columns[2]:
        metric_card("Expected surplus", f"{expected_surplus:,} meals", "Meals available after expected demand", "#e6ad45")
    with kpi_columns[3]:
        metric_card("Waste risk", f"{waste_risk:.0f}%", "Share of planned production", "#ef6f6c")

    st.markdown("### Today's recommendation")
    with st.container(border=True):
        decision_col, surplus_col = st.columns([1.2, 1])
        with decision_col:
            st.markdown('<div class="decision-title">Today\'s recommendation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="decision-value">Prepare approximately {recommended_production:,} meals today.</div>', unsafe_allow_html=True)
            if avoidable_surplus:
                st.markdown(
                    f'<div class="decision-note">Current plan: {int(planned_production):,} meals · potential avoidable surplus: {avoidable_surplus:,}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="decision-note">The current plan is aligned with the forecast buffer.</div>', unsafe_allow_html=True)
        with surplus_col:
            if expected_surplus == 0:
                st.markdown('<div class="decision-title" style="color:#70d49b;">No surplus expected</div>', unsafe_allow_html=True)
                st.markdown('<div class="decision-value">No redistribution required.</div>', unsafe_allow_html=True)
            elif redistribution_plan.empty:
                st.markdown('<div class="decision-title" style="color:#ef6f6c;">Surplus needs attention</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="decision-value">{expected_surplus:,} meals available.</div>', unsafe_allow_html=True)
                st.markdown('<div class="decision-note">No active recipient currently has available capacity.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="decision-title" style="color:#e6ad45;">Surplus can be redistributed</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="decision-value">{redistributed_meals:,} meals matched.</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="decision-note">{len(redistribution_plan)} recipient(s) · {unallocated_surplus:,} meals unmatched</div>', unsafe_allow_html=True)
                st.button(
                    "View distribution plan",
                    key="overview_distribution_plan",
                    icon=":material/arrow_forward:",
                    on_click=navigate_to,
                    args=("Recipient Matching",),
                )

    with st.expander("View production breakdown"):
        st.markdown("#### Today's production breakdown")
        render_production_breakdown(production_breakdown)

    st.markdown("### Recent demand trend")
    trend_col, comparison_col = st.columns([2.2, 1])
    with trend_col:
        with st.container(border=True):
            demand_chart = demand_trend_chart(history, model, operation_date, predicted_demand)
            if demand_chart is None:
                st.info("No historical demand data available yet.")
            else:
                st.altair_chart(demand_chart, width="stretch")
                st.caption("Green: actual production from completed outcomes. Blue dotted: stored model-predicted demand.")
    with comparison_col:
        with st.container(border=True):
            st.markdown("**Today vs recent average**")
            if history.empty:
                st.info("Historical comparison unavailable.")
            else:
                recent_history = history.assign(date=pd.to_datetime(history["date"])).sort_values("date").tail(7)
                recent_average = recent_history["actual_consumption"].mean()
                if pd.isna(recent_average) or recent_average == 0:
                    st.info("Historical comparison unavailable.")
                else:
                    average_delta = (predicted_demand - recent_average) / recent_average * 100
                    st.metric("Expected demand", f"{predicted_demand:,} meals", f"{average_delta:+.1f}%")
                    st.caption(f"Compared with {recent_average:,.0f} meals across the last {len(recent_history)} recorded day(s).")
                    if abs(average_delta) < 3:
                        st.caption("Today's expected demand is close to the recent average.")
                    elif average_delta > 0:
                        st.caption("Today's expected demand is above the recent average.")
                    else:
                        st.caption("Today's expected demand is below the recent average.")

    st.markdown("### Next actions")
    preview_col, actions_col = st.columns([1.3, .7])
    with preview_col:
        with st.container(border=True):
            st.markdown('<div class="quick-action-label">Redistribution preview</div>', unsafe_allow_html=True)
            if redistribution_plan.empty:
                st.caption("No recipient allocation is available for this scenario.")
            else:
                preview = redistribution_plan[["Recipient", "Meals allocated", "Need", "Priority", "Distance (km)"]].head(3).copy()
                preview = preview.rename(columns={"Meals allocated": "Meals"})
                st.dataframe(preview, hide_index=True, width="stretch")
                if len(redistribution_plan) > 3:
                    st.caption(f"{len(redistribution_plan) - 3} additional recipient(s) are in the full plan.")
    with actions_col:
        with st.container(border=True):
            st.markdown('<div class="quick-action-label">Quick actions</div>', unsafe_allow_html=True)
            action_row = st.columns(2)
            with action_row[0]:
                st.button(
                    "Update plan",
                    key="overview_update_plan",
                    width="stretch",
                    icon=":material/edit:",
                    on_click=navigate_to,
                    args=("Demand & Planning",),
                )
            with action_row[1]:
                st.button(
                    "Recipients",
                    key="overview_recipients",
                    width="stretch",
                    icon=":material/groups:",
                    on_click=navigate_to,
                    args=("Recipient Matching",),
                )
            action_row = st.columns(2)
            with action_row[0]:
                st.button(
                    "View map",
                    key="overview_map",
                    width="stretch",
                    icon=":material/map:",
                    on_click=navigate_to,
                    args=("Recipient Matching",),
                )
            with action_row[1]:
                st.button(
                    "Record plan",
                    key="overview_record",
                    width="stretch",
                    icon=":material/save:",
                    on_click=navigate_to,
                    args=("Recipient Matching",),
                )

elif active_section == "Demand & Planning":
    st.subheader("Planning Review")
    st.caption("Review demand forecasts, production decisions, and surplus for any available date.")

    historical_dates = pd.to_datetime(history["date"], errors="coerce").dropna() if not history.empty else pd.Series(dtype="datetime64[ns]")
    current_review_date = pd.Timestamp(default_operation["date"]).date()
    earliest_review_date = historical_dates.min().date() if not historical_dates.empty else current_review_date
    latest_review_date = max(current_review_date, historical_dates.max().date() if not historical_dates.empty else current_review_date)
    st.markdown("**Review period**")
    review_from_col, review_to_col = st.columns([1, 1])
    with review_from_col:
        review_start = st.date_input(
            "From",
            value=current_review_date,
            min_value=earliest_review_date,
            max_value=latest_review_date,
            key="planning_review_from",
        )
    with review_to_col:
        review_end = st.date_input(
            "To",
            value=current_review_date,
            min_value=earliest_review_date,
            max_value=latest_review_date,
            key="planning_review_to",
        )

    current_values = {
        "predicted_demand": predicted_demand,
        "planned_production": planned_production,
        "safety_buffer": safety_buffer,
        "holiday": holiday,
        "special_event": special_event,
        "weather": weather,
        "precipitation": precipitation,
    }
    invalid_review_period = review_start > review_end
    review_rows = pd.DataFrame()
    if not invalid_review_period:
        review_rows = planning_review_rows(
            history,
            model,
            review_start,
            review_end,
            current_review_date,
            current_values,
        )

    if invalid_review_period:
        st.error("The From date must be on or before the To date.")
    elif review_rows.empty:
        st.info("No planning data is available for the selected period.")
    else:
        is_single_date = review_start == review_end and len(review_rows) == 1
        if is_single_date:
            selected_row = review_rows.iloc[0]
            selected_date = pd.Timestamp(selected_row["date"])
            selected_prediction = int(selected_row["predicted_demand"])
            selected_recommendation = math.ceil(selected_prediction * (1 + safety_buffer / 100))
            has_actual_production = not pd.isna(selected_row["actual_production"])
            selected_production = int(
                selected_row["actual_production"] if has_actual_production else selected_row["planned_production"]
            )
            selected_surplus = (
                int(selected_row["surplus_quantity"])
                if has_actual_production
                else max(selected_production - selected_prediction, 0)
            )
            production_label = "Actual production" if has_actual_production else "Current planned production"
            surplus_label = "Actual surplus" if has_actual_production else "Expected surplus"

            st.markdown(f"### {selected_date.strftime('%d %B %Y')} · {selected_date.day_name()}")
            st.markdown(
                f"The system predicted demand for **{selected_prediction:,} meals**. "
                f"It recommended preparing **{selected_recommendation:,} meals**, while the kitchen's {production_label.lower()} was **{selected_production:,} meals**."
            )
            if not has_actual_production:
                st.caption("This date has no completed outcome; the current planned production is shown as forecast context.")
            else:
                st.caption(f"Recorded actual consumption: {int(selected_row['actual_consumption']):,} meals · actual surplus: {selected_surplus:,} meals.")

            st.markdown(
                f"""
                <div class="planning-breakdown">
                    <div class="planning-stat" style="--planning-accent:#4d9fff;">
                        <div class="planning-stat-label">Predicted demand</div>
                        <div class="planning-stat-value">{selected_prediction:,} meals</div>
                    </div>
                    <div class="planning-stat" style="--planning-accent:#50c878;">
                        <div class="planning-stat-label">Recommended production</div>
                        <div class="planning-stat-value">{selected_recommendation:,} meals</div>
                    </div>
                    <div class="planning-stat" style="--planning-accent:#9eacbf;">
                        <div class="planning-stat-label">{production_label}</div>
                        <div class="planning-stat-value">{selected_production:,} meals</div>
                    </div>
                    <div class="planning-stat" style="--planning-accent:#e6ad45;">
                        <div class="planning-stat-label">{surplus_label}</div>
                        <div class="planning-stat-value">{selected_surplus:,} meals</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            input_col, plan_col = st.columns([1.1, 1])
            with input_col:
                with st.container(border=True):
                    st.markdown("**Forecast inputs**")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Input": ["Date / weekday", "Holiday", "Special event", "Weather", "Precipitation"],
                                "Value": [
                                    f"{selected_date.strftime('%d %b %Y')} · {selected_date.day_name()}",
                                    "Yes" if selected_row["holiday"] else "No",
                                    "Yes" if selected_row["special_event"] else "No",
                                    selected_row["weather"],
                                    selected_row["precipitation"],
                                ],
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                    )
            with plan_col:
                with st.container(border=True):
                    st.markdown("**Planning variance**")
                    variance = selected_production - selected_prediction
                    direction = "above" if variance > 0 else "below" if variance < 0 else "aligned with"
                    st.markdown(f"**{variance:+,} meals**")
                    st.caption(f"{production_label} was {abs(variance):,} meals {direction} predicted demand.")
                    st.markdown("**Planning outcome**")
                    st.caption(
                        f"The system predicted {selected_prediction:,} meals and recommended {selected_recommendation:,}. "
                        f"The kitchen recorded {selected_production:,}, resulting in {surplus_label.lower()} of {selected_surplus:,} meals."
                    )

            if selected_date == pd.Timestamp(operation_date):
                st.markdown("#### Item-level production plan (current scenario)")
                render_production_breakdown(production_breakdown)
            else:
                st.caption("Item-level production breakdown is available for the current scenario date only.")
        else:
            total_predicted = int(review_rows["predicted_demand"].sum())
            completed_rows = review_rows.dropna(subset=["actual_production"])
            total_production = int(completed_rows["actual_production"].sum()) if not completed_rows.empty else 0
            aggregate_variance = total_production - int(completed_rows["predicted_demand"].sum()) if not completed_rows.empty else None
            st.markdown(f"### {pd.Timestamp(review_start).strftime('%d %b %Y')} → {pd.Timestamp(review_end).strftime('%d %b %Y')}")
            st.markdown(
                f"Across **{len(review_rows)} available record(s)**, the model predicted **{total_predicted:,} meals** "
                f"and the kitchen recorded **{total_production:,} meals** across completed outcomes."
            )
            st.caption("The range includes only dates with valid source records; missing dates are not filled in.")
            if aggregate_variance is not None:
                st.markdown(f"**Planning variance:** {aggregate_variance:+,} actual production meals versus predicted demand.")
            else:
                st.caption("No completed production outcomes are available in this range.")

        st.subheader("Plan vs Reality")
        planning_chart = planning_review_chart(review_rows)
        if planning_chart is None:
            st.info("Not enough valid data for the planning comparison.")
        else:
            st.altair_chart(planning_chart, width="stretch")
            st.caption("Blue shows predicted demand; green shows actual production from completed outcomes. Current planned production is only shown for the active scenario date.")

    with st.expander("Historical kitchen data"):
        if history.empty:
            st.info("No historical kitchen data available yet.")
        else:
            st.dataframe(history, hide_index=True, width="stretch")

elif active_section == "Raw Inventory":
    st.subheader("Raw Inventory")
    st.caption("Ingredients before production. Batches are consumed earliest-expiry-first (FEFO).")
    for message in raw_errors:
        st.warning(message)

    today_split = predicted_by_menu_item()
    notifications = raw_notification_payload(
        raw_stock, raw_today, raw_capacities, today_split, capacity_by_item
    )
    notification_total = raw_notification_count(notifications)

    # --- Notification Center: one place for stock/expiry/prediction alerts ---
    header_left, header_right = st.columns([5, 1])
    header_left.markdown("#### Notifications")
    # Separate keys: the button key and the panel flag must not collide, or
    # Streamlit rejects writing the flag after the widget is instantiated.
    if header_right.button(
        f"Notifications ({notification_total})", key="raw_notifications_toggle", width="stretch"
    ):
        st.session_state["raw_notifications_panel"] = not st.session_state.get(
            "raw_notifications_panel", False
        )
    if st.session_state.get("raw_notifications_panel"):
        with st.container(border=True):
            st.markdown("**Stock**")
            if notifications["stock"]:
                for entry in notifications["stock"]:
                    st.warning(entry)
            else:
                st.caption("No stock issues.")
            st.markdown("**Expiry Watch**")
            if notifications["expiry"]:
                for entry in notifications["expiry"]:
                    background, text_color = RAW_TONE_COLORS.get(
                        entry["tone"], RAW_TONE_COLORS["unknown"]
                    )
                    st.markdown(
                        f'<span class="status-chip" style="background:{background};'
                        f'color:{text_color};">{entry["text"]}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No expiry issues.")
            st.markdown("**Today's Prediction Feasibility**")
            if notifications["prediction"]:
                for entry in notifications["prediction"]:
                    st.warning(entry)
                st.caption(
                    "Warnings only. The demand prediction is unchanged and production is never blocked."
                )
            else:
                st.caption(
                    "Raw ingredients can support today's predicted demand."
                    if today_split
                    else "No item-level prediction is available for the selected date."
                )
            if st.button("Close notifications", key="raw_notifications_close"):
                st.session_state["raw_notifications_panel"] = False
                st.rerun()

    with st.expander("Add ingredient batch", expanded=not raw_stock.empty):
        with st.form("raw_add_form"):
            add_columns = st.columns([2, 1, 1, 1, 1])
            entered_ingredient = add_columns[0].text_input("Ingredient", value="")
            entered_quantity = add_columns[1].number_input(
                "Quantity", min_value=0.0, value=1.0, step=0.5, format="%.3f"
            )
            entered_unit = add_columns[2].selectbox("Unit", SUPPORTED_UNITS, index=0)
            added_on = add_columns[3].date_input("Date added", value=raw_today)
            expires_on = add_columns[4].date_input("Expiry date", value=raw_today)
            st.caption(
                "Canonical ingredients: " + ", ".join(RAW_INGREDIENTS) + ". "
                "Anything else is kept exactly as typed and flagged Unknown / Unmapped."
            )
            add_submitted = st.form_submit_button("Add batch", type="primary")
        if add_submitted:
            try:
                record = add_raw_batch(
                    RAW_INVENTORY_PATH, entered_ingredient, entered_quantity,
                    entered_unit, added_on, expires_on,
                )
                st.success(
                    f"{batch_label(record['date_added'], batch_sequence(record['raw_id']))} added: "
                    f"{record['ingredient']} {record['original_quantity']:g} {record['unit']}."
                )
                st.cache_data.clear()
                st.rerun()
            except RawInventoryError as mapped_error:
                try:
                    record_unmapped_batch(
                        RAW_INVENTORY_PATH, entered_ingredient, entered_quantity,
                        entered_unit, added_on, expires_on,
                    )
                    st.warning(
                        f"{mapped_error} The batch was stored unchanged and flagged "
                        f"Unknown / Unmapped — it is excluded from capacity until corrected."
                    )
                    st.cache_data.clear()
                    st.rerun()
                except RawInventoryError as unmapped_error:
                    st.error(str(unmapped_error))

    st.markdown("### Stock")
    if raw_stock.empty:
        st.info("No raw ingredient batches recorded yet. Add a batch above to start.")
    else:
        try:
            stock_view = stock_table(raw_stock, raw_today, EXPIRING_SOON_ACTIVE, APPROACHING_EXPIRY_ACTIVE)
        except (KeyError, TypeError, ValueError) as error:
            stock_view = pd.DataFrame()
            st.warning(f"Raw stock could not be displayed: {error}")
        if not stock_view.empty:
            stock_display = stock_view[
                ["batch_label", "ingredient", "canonical_ingredient", "unit",
                 "remaining_quantity", "expiry_date", "status", "days_to_expiry"]
            ].rename(columns={
                "batch_label": "Batch", "ingredient": "Entered as",
                "canonical_ingredient": "Canonical", "unit": "Unit",
                "remaining_quantity": "Remaining", "expiry_date": "Expires",
                "status": "Status", "days_to_expiry": "Days left",
            }).copy()
            stock_display["Canonical"] = (
                stock_display["Canonical"].fillna("").replace("", "None / Unmapped")
            )
            stock_display["Expires"] = pd.to_datetime(
                stock_display["Expires"], errors="coerce"
            ).dt.strftime("%d %b %Y")
            tones = stock_view["status_tone"].tolist()
            # Colour comes from the centralized tone; the Status text remains the
            # non-colour signal, so the table is never colour-dependent.
            styled = stock_display.style.apply(
                lambda row: [
                    (
                        f"color: {RAW_TONE_COLORS[tones[row.name]][1]}; "
                        f"background-color: {RAW_TONE_COLORS[tones[row.name]][0]}; "
                        "font-weight: 700;"
                    )
                    if column in ("Status", "Days left") else ""
                    for column in stock_display.columns
                ],
                axis=1,
            )
            st.dataframe(styled, hide_index=True, width="stretch")
            st.caption(
                "Canonical shows whether the entered ingredient matches one of the seven "
                "canonical ingredients. 'None / Unmapped' means it still needs correction."
            )
            st.caption(
                "Colour key: dark = Expired · red = expires very soon · "
                "yellow = approaching expiry · green = plenty of time remaining."
            )

    if not stock_view.empty:
        st.markdown("#### Batch management")
        st.caption("Select any batch to correct its details or remove it.")
        batch_options = {
            str(row["raw_id"]): (
                f"{row['batch_label']} - {row['ingredient']} - "
                f"{float(row['remaining_quantity']):g} {row['unit']}"
            )
            for _, row in stock_view.iterrows()
        }
        selected_id = st.selectbox(
            "Batch", list(batch_options.keys()),
            format_func=lambda value: batch_options[value],
            key="raw_manage_batch",
        )
        selected_row = stock_view[stock_view["raw_id"].astype(str) == selected_id].iloc[0]
        st.caption(
            f"Selected: {batch_options[selected_id]} · internal id {selected_id} "
            "(used for storage, FEFO and usage history only)."
        )
        manage_action = st.radio(
            "Action", ["Edit", "Remove"], horizontal=True, key="raw_manage_action"
        )
        if manage_action == "Edit":
            with st.form("raw_edit_form"):
                edit_columns = st.columns([2, 1, 1, 1, 1])
                edit_ingredient = edit_columns[0].text_input(
                    "Entered as", value=str(selected_row["ingredient"])
                )
                edit_quantity = edit_columns[1].number_input(
                    "Quantity", min_value=0.0,
                    value=float(selected_row["original_quantity"] or 0.0),
                    step=0.5, format="%.3f",
                )
                edit_unit = edit_columns[2].selectbox(
                    "Unit", SUPPORTED_UNITS,
                    index=(
                        SUPPORTED_UNITS.index(str(selected_row["unit"]))
                        if str(selected_row["unit"]) in SUPPORTED_UNITS else 0
                    ),
                )
                edit_added = edit_columns[3].date_input(
                    "Date added", value=pd.Timestamp(selected_row["date_added"]).date()
                )
                edit_expires = edit_columns[4].date_input(
                    "Expiry date", value=pd.Timestamp(selected_row["expiry_date"]).date()
                )
                st.caption(
                    "Correcting the ingredient to a canonical name (e.g. 'chickem' to "
                    "'chicken') makes the batch immediately usable in recipes and capacity."
                )
                edit_submitted = st.form_submit_button("Save changes")
            if edit_submitted:
                try:
                    updated = update_raw_batch(
                        RAW_INVENTORY_PATH, selected_id, edit_ingredient,
                        edit_quantity, edit_unit, edit_added, edit_expires,
                    )
                    st.cache_data.clear()
                    if updated["canonical_ingredient"]:
                        st.success(
                            f"Batch updated: '{updated['ingredient']}' now maps to "
                            f"'{updated['canonical_ingredient']}'."
                        )
                    else:
                        st.warning(
                            f"'{updated['ingredient']}' is not one of the seven canonical "
                            "ingredients. The batch was saved unchanged and is excluded "
                            "from recipe and capacity calculations until corrected."
                        )
                    st.rerun()
                except RawInventoryError as edit_error:
                    st.error(str(edit_error))
        else:
            st.warning(
                f"This removes {batch_options[selected_id]}. Historical usage records that "
                "already reference this batch are kept."
            )
            confirm = st.checkbox("Yes, remove this batch", key="raw_remove_confirm")
            if st.button("Remove batch", key="raw_remove_btn", disabled=not confirm):
                try:
                    remove_raw_batch(RAW_INVENTORY_PATH, selected_id)
                    st.session_state.pop("raw_manage_batch", None)
                    st.session_state.pop("raw_manage_action", None)
                    st.session_state.pop("raw_remove_confirm", None)
                    st.cache_data.clear()
                    st.rerun()
                except RawInventoryError as remove_error:
                    st.error(str(remove_error))

    st.markdown("### Production capacity")
    st.caption(
        "This table shows how many servings of each menu item can be made from the "
        "ingredients currently remaining in raw inventory."
    )
    st.caption(
        "Recipe quantities are prototype demo values, not food-safety, nutrition, "
        "or industry standards."
    )
    if not raw_capacities:
        st.warning("Production capacity is unavailable for now.")
    else:
        capacity_rows = [
            {
                "Menu item": entry["menu_item"],
                "How many servings": entry["capacity"],
                "Bottleneck": entry["bottleneck"] or "-",
                "Ingredient support": " · ".join(
                    f"{name} {info['servings']:,}"
                    for name, info in entry["ingredient_support"].items()
                ) if entry["ingredient_support"] else "-",
            }
            for entry in raw_capacities
        ]
        st.dataframe(pd.DataFrame(capacity_rows), hide_index=True, width="stretch")
        for entry in raw_capacities:
            if entry["issue"]:
                st.caption(f"{entry['menu_item']}: {entry['issue']}")

    st.markdown("### Baseline preparedness")
    st.caption(
        "This compares your current production capacity with the recommended baseline "
        "needed for normal operations, based on recent demand plus a safety buffer."
    )
    st.caption(
        f"Recommended baseline = average demand over the last "
        f"{int(kitchen_settings['baseline_window_days'])} completed days + "
        f"{float(kitchen_settings['safety_buffer_pct']):g}% safety buffer."
    )
    if not raw_readiness:
        st.info("Baseline readiness is unavailable for now.")
    else:
        st.dataframe(
            pd.DataFrame([
                {
                    "Menu item": row["menu_item"],
                    "Current capacity": row["current_capacity"],
                    "Recommended baseline": f"{row['recommended_baseline']:,.0f}",
                    "Status": row["status"],
                    "Shortfall": f"{-row['gap']:,.0f}" if row["gap"] > 0 else "-",
                }
                for row in raw_readiness
            ]),
            hide_index=True,
            width="stretch",
        )
        # Baseline warnings stay visible here on purpose: this is an operational
        # status, so it is deliberately not moved into the Notification Center.
        below = [row for row in raw_readiness if row["gap"] > 0]
        if below:
            for row in below:
                st.info(
                    f"{row['menu_item']} is below the recommended baseline by "
                    f"{-row['gap']:,.0f} servings ({row['current_capacity']:,} available "
                    f"vs {row['recommended_baseline']:,.0f} recommended)."
                )
        else:
            st.success("All menu items are at or above the recommended baseline.")

    with st.expander("Recipe / bill of materials"):
        st.dataframe(
            pd.DataFrame([
                {
                    "Menu item": item,
                    "Ingredient": name,
                    "Per serving": f"{amount:g} {ingredient_unit(name)}",
                }
                for item in MENU_ITEMS
                for name, amount in (recipe_for(item) or [])
            ]),
            hide_index=True,
            width="stretch",
        )
        st.caption(PROTOTYPE_NOTE)

    with st.expander("Raw usage history"):
        try:
            usage = load_raw_usage(RAW_USAGE_PATH)
        except (OSError, ValueError, pd.errors.ParserError):
            usage = pd.DataFrame()
            st.warning("Raw usage history could not be read.")
        if usage.empty:
            st.info("No raw ingredient usage recorded yet.")
        else:
            st.dataframe(
                usage.sort_values("date", ascending=False), hide_index=True, width="stretch"
            )

elif active_section == "Recipient Matching":
    st.subheader("Recipient Matching")
    st.caption("Review the system's automatic recommendation, then approve it or take control.")
    selected_batch = st.session_state.get("selected_inventory_item") or {}
    selected_inventory_id = str(selected_batch.get("inventory_id", "") or "")
    selected_menu_item = str(selected_batch.get("menu_item", "") or "")
    if selected_inventory_id and not inventory.empty:
        live_match = inventory[inventory["inventory_id"].astype(str) == selected_inventory_id]
        if not live_match.empty:
            selected_menu_item = str(live_match.iloc[0]["menu_item"])
            selected_batch = {
                "inventory_id": selected_inventory_id,
                "menu_item": selected_menu_item,
                "remaining_quantity": int(live_match.iloc[0]["remaining_quantity"]),
            }
            st.session_state["selected_inventory_item"] = selected_batch
    if selected_inventory_id:
        st.info(
            f"Selected surplus — **{selected_menu_item}**: "
            f"{int(selected_batch.get('remaining_quantity', 0)):,} meals available "
            f"(batch {selected_inventory_id})."
        )
        matching_surplus = int(selected_batch.get("remaining_quantity", 0))
        matching_menu_item = selected_menu_item
    else:
        # Redistribute-All context: multiple item batches prepared as one review.
        all_context = st.session_state.get("redistribute_all_items") or []
        if all_context:
            matching_surplus = int(sum(int(b.get("remaining_quantity", 0)) for b in all_context))
            matching_menu_item = "__ALL__"
            st.info(
                f"Redistribute All — **{len(all_context)} item batch(es)**, "
                f"{matching_surplus:,} meals total. Items stay separate."
            )
        else:
            matching_surplus, matching_menu_item = expected_surplus, ""
    if selected_inventory_id:
        qty_key = f"qty_{selected_inventory_id}"
    else:
        qty_key = "qty_all"
    qty_default = int(st.session_state.get(qty_key, matching_surplus))
    qty_default = max(0, min(qty_default, matching_surplus))
    quantity_to_move = st.number_input(
        "Quantity to redistribute",
        min_value=0,
        max_value=int(matching_surplus),
        value=int(qty_default),
        step=1,
        key=qty_key,
    )
    strategy = st.selectbox(
        "Distribution strategy",
        ["Even Distribution", "Manual Allocation", "Item-wise Allocation"],
        index=0,
    )
    # Selected surplus context (item batches). This is context only: recipient-level
    # strategies never create one allocation per batch.
    if selected_inventory_id:
        target_batches = [
            {
                "inventory_id": selected_inventory_id,
                "menu_item": matching_menu_item,
                "remaining_quantity": int(matching_surplus),
                "qty": int(quantity_to_move),
            }
        ]
    elif matching_menu_item == "__ALL__":
        target_batches = [
            {
                "inventory_id": b.get("inventory_id"),
                "menu_item": b.get("menu_item"),
                "remaining_quantity": int(b.get("remaining_quantity", 0)),
                "qty": int(b.get("remaining_quantity", 0)),
            }
            for b in (st.session_state.get("redistribute_all_items") or [])
        ]
    else:
        target_batches = []
    recip_rows = recipients[recipients["active"] == 1].to_dict("records")
    headroom = {r["recipient_id"]: min(int(r["current_need"]), int(r["capacity"])) for r in recip_rows}
    item_names = []
    item_available = {}
    for b in target_batches:
        name = str(b["menu_item"])
        if name not in item_names:
            item_names.append(name)
        item_available[name] = item_available.get(name, 0) + int(b["remaining_quantity"])
    total_available = sum(int(b["qty"]) for b in target_batches)
    item_issues = []
    if strategy == "Item-wise Allocation" and item_names:
        st.markdown("**Item-wise allocation** — one row per recipient, one column per selected item.")
        st.caption(" · ".join(f"{n} {item_available[n]:,} available" for n in item_names))
        editor_rows = pd.DataFrame(
            [{"Recipient": r["name"], **{n: 0 for n in item_names}} for r in recip_rows]
        )
        edited = st.data_editor(
            editor_rows,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            disabled=["Recipient"],
            column_config={
                n: st.column_config.NumberColumn(n, min_value=0, step=1, format="%d")
                for n in item_names
            },
            key=f"itemwise_{qty_key}",
        )
        matrix = []
        for position, r in enumerate(recip_rows):
            values = {}
            for name in item_names:
                raw = edited.iloc[position][name]
                value = 0 if pd.isna(raw) else int(raw)
                values[name] = max(value, 0)
            matrix.append(values)
            total_row = sum(values.values())
            limit = int(headroom.get(r["recipient_id"], 0))
            if total_row > limit:
                item_issues.append(
                    f"{r['name']} can receive a maximum of {limit:,} meals. "
                    f"Current allocation: {total_row:,}."
                )
        for name in item_names:
            allocated = sum(row[name] for row in matrix)
            if allocated > item_available[name]:
                item_issues.append(
                    f"{name} allocation exceeds available quantity: "
                    f"{allocated:,} requested, {item_available[name]:,} available."
                )
        rows = []
        for r, values in zip(recip_rows, matrix):
            for name in item_names:
                if values[name] <= 0:
                    continue
                rows.append(
                    {
                        "recipient_id": r["recipient_id"],
                        "Recipient": r["name"],
                        "Meals allocated": values[name],
                        "Need": int(r["current_need"]),
                        "Priority": r["priority"],
                        "Distance (km)": float(r["distance_km"]),
                        "latitude": float(r["latitude"]),
                        "longitude": float(r["longitude"]),
                        "menu_item": name,
                    }
                )
        redistribution_plan = pd.DataFrame(
            rows,
            columns=[
                "recipient_id", "Recipient", "Meals allocated", "Need",
                "Priority", "Distance (km)", "latitude", "longitude", "menu_item",
            ],
        )
        unallocated_surplus = max(
            total_available - int(redistribution_plan["Meals allocated"].sum()), 0
        )
        for issue in item_issues:
            st.error(issue)
        if not item_issues:
            totals_table = pd.DataFrame(
                [
                    {
                        "Item": name,
                        "Available": item_available[name],
                        "Allocated": sum(row[name] for row in matrix),
                    }
                    for name in item_names
                ]
            )
            totals_table["Remaining"] = (
                totals_table["Available"] - totals_table["Allocated"]
            ).clip(lower=0)
            st.dataframe(totals_table, hide_index=True, width="stretch")
    elif strategy == "Manual Allocation" and target_batches:
        st.markdown("**Manual allocation** — total meals per recipient:")
        st.caption(
            f"{total_available:,} meals available across {len(item_names)} selected item(s)."
        )
        manual_rows = []
        for r in recip_rows:
            cap = int(headroom.get(r["recipient_id"], 0))
            qty = st.number_input(
                f"{r['name']} — meals (max {cap:,})",
                min_value=0,
                max_value=max(cap, 0),
                value=0,
                step=1,
                key=f"manual_{qty_key}_{r['recipient_id']}",
            )
            if int(qty) > 0:
                manual_rows.append(
                    {
                        "recipient_id": r["recipient_id"],
                        "Recipient": r["name"],
                        "Meals allocated": int(qty),
                        "Need": int(r["current_need"]),
                        "Priority": r["priority"],
                        "Distance (km)": float(r["distance_km"]),
                        "latitude": float(r["latitude"]),
                        "longitude": float(r["longitude"]),
                    }
                )
        redistribution_plan = pd.DataFrame(
            manual_rows,
            columns=[
                "recipient_id", "Recipient", "Meals allocated", "Need",
                "Priority", "Distance (km)", "latitude", "longitude",
            ],
        )
        manual_total = int(redistribution_plan["Meals allocated"].sum())
        unallocated_surplus = max(total_available - manual_total, 0)
        if manual_total > total_available:
            item_issues.append(
                f"Only {total_available:,} meals are available, but "
                f"{manual_total:,} meals were allocated."
            )
            st.error(
                f"Only {total_available:,} meals are available, but "
                f"{manual_total:,} meals were allocated."
            )
    else:
        sub_rec = recipients.copy()
        # Even Distribution is recipient-level: one allocation per unique recipient.
        sub_rec["current_need"] = sub_rec["recipient_id"].map(headroom).fillna(0).astype(int)
        sub_rec["capacity"] = sub_rec["recipient_id"].map(headroom).fillna(0).astype(int)
        redistribution_plan, unallocated_surplus = build_even_plan(
            int(quantity_to_move), sub_rec
        )
    if redistribution_plan.empty:
        redistribution_plan = pd.DataFrame(
            columns=[
                "recipient_id", "Recipient", "Meals allocated", "Need",
                "Priority", "Distance (km)", "latitude", "longitude",
            ]
        )
    redistributed_meals = int(redistribution_plan["Meals allocated"].sum()) if not redistribution_plan.empty else 0
    if "menu_item" in redistribution_plan.columns:
        item_breakdown = redistribution_plan
    else:
        item_breakdown = split_plan_by_batch(redistribution_plan, target_batches)
    # Recipient-level view: each unique recipient appears exactly once.
    recipient_level = redistribution_plan
    if "menu_item" in redistribution_plan.columns:
        recipient_level = (
            redistribution_plan.groupby(
                ["recipient_id", "Recipient", "Need", "Priority", "Distance (km)", "latitude", "longitude"],
                as_index=False,
            )["Meals allocated"]
            .sum()
            .sort_values("Meals allocated", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
    recipient_count = len(recipient_level)
    with st.container(border=True):
        summary_columns = st.columns(4)
        summary_columns[0].metric("Available surplus", f"{matching_surplus:,} meals")
        summary_columns[1].metric("Recipients matched", f"{recipient_count}")
        summary_columns[2].metric("Meals allocated", f"{redistributed_meals:,}")
        summary_columns[3].metric("Unallocated", f"{unallocated_surplus:,}")

    if matching_surplus == 0:
        st.success("No redistribution required.")
    elif redistribution_plan.empty:
        st.warning("Surplus available, but no suitable recipient is currently matched.")
    elif unallocated_surplus:
        st.warning(f"{redistributed_meals:,} surplus meals allocated; {unallocated_surplus:,} meals remain unallocated.")
    else:
        st.success("Surplus ready for redistribution.")

    if not redistribution_plan.empty:
        if selected_inventory_id:
            st.caption(f"Item context: {matching_menu_item} · batch {selected_inventory_id}.")
        st.subheader("Matched recipients")
        recipient_columns = st.columns(min(recipient_count, 3))
        priority_colors = {"High": "#ef6f6c", "Medium": "#e6ad45", "Low": "#70d49b"}
        for index, recipient in enumerate(recipient_level.to_dict("records")):
            with recipient_columns[index % len(recipient_columns)]:
                with st.container(border=True):
                    priority_color = priority_colors.get(recipient["Priority"], "#9eacbf")
                    st.markdown(f"**{recipient['Recipient']}**")
                    st.markdown(
                        f'<span class="status-chip" style="color:{priority_color};">{recipient["Priority"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.metric("Allocated", f"{int(recipient['Meals allocated']):,} meals")
                    st.caption(f"Need {int(recipient['Need']):,} · {float(recipient['Distance (km)']):.1f} km")

        allocation_view = recipient_level[
            ["Recipient", "Need", "Meals allocated", "Priority", "Distance (km)"]
        ].rename(columns={"Meals allocated": "Allocated meals"})
        allocation_view["Allocation status"] = "Matched"
        allocation_style = allocation_view.style.apply(
            lambda column: [
                f"color: {priority_colors.get(value, '#9eacbf')}; font-weight: 700;"
                for value in column
            ],
            subset=["Priority"],
        )
        st.dataframe(allocation_style, hide_index=True, width="stretch")
        st.caption("Priority order: High, Medium, Low. Allocations remain limited by each recipient's need and capacity.")

        if not item_breakdown.empty:
            pivot = item_breakdown.pivot_table(
                index="Recipient", columns="menu_item", values="Meals allocated", aggfunc="sum", fill_value=0
            )
            pivot["Total"] = pivot.sum(axis=1)
            st.markdown("**Item-wise allocation (recipient × item)**")
            st.dataframe(pivot.reset_index(), hide_index=True, width="stretch")

        st.subheader("Redistribution plan")
        st.table(
            {
                "Route": "Kitchen → " + " → ".join(recipient_level["Recipient"].unique()),
                "Total distance": f"{recipient_level['Distance (km)'].sum():.1f} km",
                "Meals in transit": f"{redistributed_meals:,}",
            },
            border="horizontal",
            width="content",
        )

    if not redistribution_plan.empty:
        st.subheader("Map / route plan")
        with st.container(border=True):
            st.pydeck_chart(build_route_map(redistribution_plan), width="stretch")
            st.caption("The map shows the planned dispatch sequence and recipient locations; it is not road-network navigation.")

    if target_batches:
        with st.container(border=True):
            st.markdown("**Ready to redistribute?** Review the allocation above, then save.")
            save_clicked = st.button("Save Redistribution", type="primary", key="save_redistribution")
        if save_clicked:
            if item_issues:
                st.error("Fix the allocation errors above before saving.")
            elif redistributed_meals <= 0:
                st.error("No meals have been allocated yet. Enter at least one quantity before saving.")
            elif selected_inventory_id or matching_menu_item == "__ALL__":
                final_plan = item_breakdown
                final_total = int(redistributed_meals)
                saved_key = (
                    f"redistribution_saved_{selected_inventory_id or 'ALL'}_"
                    f"{operation_date.isoformat()}_{strategy}_{final_total}_"
                    f"{'_'.join(sorted(final_plan['recipient_id'].astype(str) + '_' + final_plan['menu_item'].astype(str))) if not final_plan.empty else 'none'}"
                )
                if st.session_state.get(saved_key):
                    st.info("This redistribution was already recorded; inventory was not decremented again.")
                else:
                    try:
                        by_batch = {}
                        for b in target_batches:
                            sub = final_plan[final_plan["menu_item"] == b["menu_item"]]
                            by_batch[b["inventory_id"]] = (b, int(sub["Meals allocated"].sum()) if not sub.empty else 0, sub)
                        done = 0
                        for bid, (b, qty, sub) in by_batch.items():
                            if qty <= 0 or sub.empty:
                                continue
                            append_plan_records(sub, operation_date.isoformat(), b["menu_item"])
                            new_remaining = apply_inventory_decrement(DATA_DIR / "inventory.csv", bid, qty)
                            done += qty
                            if bid == selected_inventory_id:
                                st.session_state["selected_inventory_item"] = {
                                    "inventory_id": bid,
                                    "menu_item": b["menu_item"],
                                    "remaining_quantity": int(new_remaining),
                                }
                        if matching_menu_item == "__ALL__":
                            st.session_state.pop("redistribute_all_items", None)
                            st.session_state.pop("selected_inventory_item", None)
                        left = int(matching_surplus) - done
                        st.session_state[saved_key] = True
                        st.success(f"{done:,} meals allocated successfully. {left:,} meals remain in surplus inventory.")
                        st.cache_data.clear()
                    except ValueError as error:
                        st.error(f"Redistribution was not saved: {error}")
    elif not matching_surplus:
        st.info("Select a surplus batch from Surplus Inventory to record an item-level redistribution.")

    with st.expander("All active recipients"):
        active_recipients = recipients[recipients["active"] == 1]
        if active_recipients.empty:
            st.info("No suitable recipients currently available.")
        else:
            st.dataframe(
                active_recipients[["name", "type", "current_need", "capacity", "priority", "distance_km"]],
                hide_index=True,
                width="stretch",
            )

elif active_section == "Impact & Analytics":
    st.markdown("## Impact & Analytics")
    st.caption("Completed redistribution only. Planned records never count as impact.")
    period = st.segmented_control("Period", ["7 Days", "30 Days", "90 Days", "All Time"], default="30 Days")
    end_day = pd.Timestamp.today().date()
    if not records.empty:
        end_day = pd.to_datetime(records["date"], errors="coerce").max()
        end_day = end_day.date() if pd.notna(end_day) else pd.Timestamp.today().date()
    start_day = impact_period_cutoff(period, end_day)
    delivered_all = records[records["status"].str.lower() == "delivered"].copy()
    delivered = impact_filter_by_period(delivered_all, "date", start_day)
    p_history = impact_filter_by_period(history, "date", start_day)
    p_items = impact_filter_by_period(item_history, "date", start_day)
    meals = int(delivered["allocated_quantity"].sum()) if not delivered.empty else 0
    n_recip = int(delivered["recipient_id"].nunique()) if not delivered.empty else 0
    n_deliv = int(len(delivered))
    co2e = meals * 0.5
    st.markdown("##### IMPACT THIS PERIOD")
    hero_l, hero_r = st.columns([1.6, 1])
    with hero_l:
        st.markdown(f"# {meals:,} ")
        st.markdown("### MEALS REDISTRIBUTED")
    with hero_r:
        st.metric("Recipients served", f"{n_recip}")
        st.metric("Completed redistributions", f"{n_deliv}")
        st.metric("Est. CO2e avoided", f"{co2e:,.1f} kg")
    st.divider()
    st.markdown("##### Redistribution Over Time")
    time_mode = st.segmented_control("View", ["Specific Date", "All Time"], default="All Time")
    if time_mode == "Specific Date":
        avail = sorted(pd.to_datetime(records["date"], errors="coerce").dropna().dt.date.unique()) if not records.empty else []
        sel = st.date_input("Date", value=avail[-1] if avail else end_day, min_value=min(avail) if avail else None, max_value=max(avail) if avail else None)
        day_df = delivered_all[pd.to_datetime(delivered_all["date"], errors="coerce").dt.date == sel] if not delivered_all.empty else delivered_all.head(0)
        if day_df.empty:
            st.info(f"No completed redistribution on {sel}.")
        else:
            day_agg = day_df.groupby("recipient_id", as_index=False)["allocated_quantity"].sum().merge(recipients[["recipient_id", "name"]], on="recipient_id", how="left")
            st.bar_chart(day_agg.set_index("name")[["allocated_quantity"]], height=260)
            st.dataframe(day_agg.rename(columns={"name": "Recipient", "allocated_quantity": "Meals"}), hide_index=True, width="stretch")
    else:
        if delivered_all.empty:
            st.info("No completed redistributions recorded yet.")
        else:
            trend = (
                delivered_all.assign(date=pd.to_datetime(delivered_all["date"], errors="coerce").dt.normalize())
                .groupby("date", as_index=False)["allocated_quantity"].sum().sort_values("date")
            )
            trend["date"] = trend["date"].dt.strftime("%b %d")
            st.bar_chart(trend.set_index("date")[["allocated_quantity"]], height=260)
            st.caption("All-time completed meals per date, aggregated from Delivered records.")
    st.divider()
    generated = int(p_history["surplus_quantity"].sum()) if not p_history.empty else 0
    redistributed_ref = int(delivered_all["allocated_quantity"].sum())
    remaining_now = int(inventory["remaining_quantity"].sum()) if not inventory.empty else 0
    rate = (redistributed_ref / generated * 100) if generated > 0 else 0.0
    st.markdown("##### Surplus journey")
    j1, j2, j3, j4 = st.columns(4)
    j1.metric("Generated", f"{generated:,}")
    j2.metric("Redistributed", f"{redistributed_ref:,}")
    j3.metric("In inventory", f"{remaining_now:,}")
    j4.metric("Rescue rate", f"{rate:.0f}%")
    st.progress(min(max(rate / 100, 0.0), 1.0))
    st.caption("Redistributed surplus / total surplus generated.")
    st.divider()
    st.markdown("##### Food items")
    if p_items.empty:
        st.info("No item-level data in this period.")
    else:
        item_gen = p_items.groupby("menu_item", as_index=False)["surplus_quantity"].sum().sort_values("surplus_quantity", ascending=False)
        d_all = delivered_all.copy()
        if "menu_item" not in d_all.columns:
            d_all["menu_item"] = ""
        matched = d_all[d_all["menu_item"].astype(str).str.strip() != ""]
        matched_p = matched[pd.to_datetime(matched["date"], errors="coerce").dt.date >= start_day] if start_day is not None and not matched.empty else matched
        unmatched_p = delivered[delivered["menu_item"].astype(str).str.strip() == ""] if "menu_item" in delivered.columns and not delivered.empty else delivered.head(0)
        unmatched_qty = int(unmatched_p["allocated_quantity"].sum()) if not unmatched_p.empty else 0
        item_red = matched_p.groupby("menu_item", as_index=False)["allocated_quantity"].sum() if not matched_p.empty else pd.DataFrame(columns=["menu_item", "allocated_quantity"])
        item_view = item_gen.merge(item_red, on="menu_item", how="left").fillna({"allocated_quantity": 0})
        item_view["redistributed"] = item_view["allocated_quantity"].astype(int)
        item_view["remaining"] = (item_view["surplus_quantity"].astype(int) - item_view["redistributed"]).clip(lower=0)
        st.bar_chart(item_view.set_index("menu_item")[["surplus_quantity"]], height=240)
        st.dataframe(item_view[["menu_item", "surplus_quantity", "redistributed", "remaining"]].rename(columns={"menu_item": "Item", "surplus_quantity": "Generated", "redistributed": "Redistributed", "remaining": "Remaining"}), hide_index=True, width="stretch")
        if unmatched_qty > 0:
            st.caption(f"{unmatched_qty:,} Delivered meal(s) predate item tracking and cannot be assigned to an item.")
    st.divider()
    st.markdown("##### Estimated CO2e avoided")
    st.markdown(f"# {co2e:,.0f} kg")
    st.caption("Estimated CO2e avoided. Estimate based on the project's emission factor (0.5 kg per meal); actual impact varies by food type and lifecycle.")
    st.divider()
    st.markdown("##### System insights")
    insights = []
    if not p_items.empty:
        top = p_items.groupby("menu_item")["surplus_quantity"].sum().sort_values(ascending=False)
        insights.append(f"{top.index[0]} generated the largest share of surplus this period ({int(top.iloc[0]):,} meals).")
    insights.append(f"{rate:.0f}% of generated surplus was redistributed.")
    unalloc = max(generated - redistributed_ref, 0)
    insights.append(f"{unalloc:,} meals of generated surplus remain unallocated overall.")
    insights.append(f"{n_recip} recipient institutions received surplus food in this period.")
    for insight in insights[:4]:
        st.info(insight)
    with st.expander("Historical redistribution records"):
        if records.empty:
            st.info("No redistribution records available yet.")
        else:
            st.dataframe(records.sort_values("date", ascending=False), hide_index=True, width="stretch")

    st.divider()
    st.markdown("##### Raw Ingredient Usage")
    st.caption("Historical raw ingredient consumption recorded from confirmed production.")
    try:
        raw_usage_all = load_raw_usage(RAW_USAGE_PATH)
    except (OSError, ValueError, pd.errors.ParserError):
        raw_usage_all = pd.DataFrame()
    # Reuse the same period cutoff already applied to the rest of this page.
    period_usage = usable_usage(raw_usage_all, start_day, end_day)
    if period_usage.empty:
        st.info("No raw ingredient usage recorded for this period.")
    else:
        unit_totals = usage_totals_by_unit(period_usage)
        ingredient_rows = usage_by_ingredient(period_usage)
        menu_rows = usage_by_menu_item(period_usage)
        weight = unit_totals[unit_totals["Unit"] == "kg"]["Total used"].sum()
        liquid = unit_totals[unit_totals["Unit"] == "L"]["Total used"].sum()
        other = unit_totals[~unit_totals["Unit"].isin(["kg", "L"])]["Total used"].sum()

        st.markdown("**Usage summary**")
        sum_cols = st.columns(3)
        sum_cols[0].metric("Weight used", f"{weight:,.2f} kg" if weight else "0 kg")
        sum_cols[1].metric("Liquid used", f"{liquid:,.2f} L" if liquid else "0 L")
        sum_cols[2].metric("Ingredients used", f"{period_usage['ingredient'].nunique()}")
        st.caption(
            f"{period_usage['menu_item'].nunique()} menu item(s) consumed raw ingredients. "
            "Weight and liquid are reported separately and never combined."
        )
        if other:
            st.caption(f"Other units: {other:,.2f} (see the unit totals below).")

        st.markdown("**Usage by ingredient**")
        st.bar_chart(ingredient_rows.set_index("Ingredient")[["Quantity used"]], height=240)
        st.dataframe(ingredient_rows, hide_index=True, width="stretch")

        st.markdown("**Most-used ingredients**")
        st.dataframe(ingredient_rows.head(5), hide_index=True, width="stretch")
        st.caption("Ranked by quantity used across the selected period, per unit.")

        st.markdown("**Usage by menu item**")
        st.dataframe(menu_rows, hide_index=True, width="stretch")

        with st.expander("Usage over time"):
            available_units = sorted(period_usage["unit"].unique().tolist())
            chosen_unit = st.selectbox(
                "Unit", available_units, key="raw_usage_trend_unit"
            )
            trend = usage_over_time(period_usage, chosen_unit)
            if trend.empty:
                st.info(f"No {chosen_unit} usage recorded for this period.")
            else:
                chart = trend.assign(
                    date=trend["date"].dt.strftime("%d %b %Y")
                )
                st.bar_chart(chart.set_index("date")[["Quantity used"]], height=240)
                st.caption(
                    f"Daily {chosen_unit} usage. Units are charted separately so kg and "
                    "litres are never mixed on one axis."
                )

        with st.expander("Raw usage records"):
            st.dataframe(
                period_usage.sort_values("date", ascending=False)[
                    ["date", "menu_item", "ingredient", "quantity_used", "unit"]
                ].rename(columns={
                    "date": "Date", "menu_item": "Menu item", "ingredient": "Ingredient",
                    "quantity_used": "Quantity used", "unit": "Unit",
                }),
                hide_index=True,
                width="stretch",
            )

elif active_section == "End of the Day":
    st.subheader("End of the Day")
    st.caption("Record today's actual operational outcome after the kitchen has finished service.")
    saved_message = st.session_state.pop("end_of_day_saved_message", None)
    if saved_message:
        st.success(saved_message)
    current_context_key = (
        str(operation_date), int(predicted_demand),
        int(holiday), int(special_event), weather, precipitation, float(safety_buffer),
        tuple(selected_menu_items),
    )

    st.markdown("### Operation context")
    context_table = pd.DataFrame(
        {
            "Context": [
                "Date", "Predicted demand", "Recommended production", "Safety buffer",
            ],
            "Value": [
                f"{pd.Timestamp(operation_date).strftime('%d %b %Y')} · {pd.Timestamp(operation_date).day_name()}",
                f"{predicted_demand:,} meals",
                f"{recommended_production:,} meals",
                f"{safety_buffer:.1f}%",
            ],
        }
    )
    st.dataframe(context_table, hide_index=True, width="stretch")

    st.markdown("### Actual results")
    if production_breakdown is None:
        st.warning(
            "No menu items are selected for today. Choose today's menu items in the "
            "Kitchen scenario to record end-of-day results."
        )
        calculate_outcome = False
    else:
        item_plan_rows = production_breakdown[production_breakdown["Menu Item"] != "TOTAL"]
        with st.form("end_of_day_form"):
            header_columns = st.columns([2.2, 1, 1, 1])
            header_columns[0].markdown("**Menu Item**")
            header_columns[1].markdown("**Actual Production**")
            header_columns[2].markdown("**Actual Consumption**")
            header_columns[3].markdown("**Surplus**")

            item_input_rows = []
            for _, plan_row in item_plan_rows.iterrows():
                item_name = plan_row["Menu Item"]
                row_columns = st.columns([2.2, 1, 1, 1])
                row_columns[0].markdown(f"**{item_name}**")
                item_production = row_columns[1].number_input(
                    f"Actual production · {item_name}",
                    min_value=0,
                    value=int(plan_row["Recommended"]),
                    step=1,
                    key=f"eod_production_{item_name}",
                    label_visibility="collapsed",
                )
                item_consumption = row_columns[2].number_input(
                    f"Actual consumption · {item_name}",
                    min_value=0,
                    value=min(int(plan_row["Predicted"]), int(plan_row["Recommended"])),
                    step=1,
                    key=f"eod_consumption_{item_name}",
                    label_visibility="collapsed",
                )
                item_surplus = int(item_production) - int(item_consumption)
                row_columns[3].markdown(f"{item_surplus:,}")
                item_input_rows.append(
                    {
                        "menu_item": item_name,
                        "predicted_quantity": int(plan_row["Predicted"]),
                        "actual_production": int(item_production),
                        "actual_consumption": int(item_consumption),
                        "surplus_quantity": item_surplus,
                    }
                )

            total_production = sum(row["actual_production"] for row in item_input_rows)
            total_consumption = sum(row["actual_consumption"] for row in item_input_rows)
            total_surplus = total_production - total_consumption
            total_columns = st.columns([2.2, 1, 1, 1])
            total_columns[0].markdown("**TOTAL**")
            total_columns[1].markdown(f"**{total_production:,}**")
            total_columns[2].markdown(f"**{total_consumption:,}**")
            total_columns[3].markdown(f"**{total_surplus:,}**")
            calculate_outcome = st.form_submit_button("Calculate outcome", type="primary")

    if calculate_outcome:
        invalid_items = [
            row["menu_item"] for row in item_input_rows
            if row["actual_consumption"] > row["actual_production"]
        ]
        if not item_input_rows:
            st.error("No menu items are selected for today.")
            st.session_state.pop("end_of_day_result", None)
        elif invalid_items:
            st.error(
                "Actual consumption cannot exceed actual production for: "
                + ", ".join(invalid_items)
                + "."
            )
            st.session_state.pop("end_of_day_result", None)
        else:
            st.session_state["end_of_day_result"] = {
                "date": operation_date,
                "predicted_demand": int(predicted_demand),
                "recommended_production": int(recommended_production),
                "actual_production": int(total_production),
                "actual_consumption": int(total_consumption),
                "holiday": int(holiday),
                "special_event": int(special_event),
                "weather": weather,
                "precipitation": precipitation,
                "safety_buffer": float(safety_buffer),
                "context_key": current_context_key,
                "item_rows": item_input_rows,
            }

    result = st.session_state.get("end_of_day_result")
    if result and result.get("context_key") != current_context_key:
        st.session_state.pop("end_of_day_result", None)
        result = None
    if result:
        actual_surplus = result["actual_production"] - result["actual_consumption"]
        prediction_deviation = result["actual_consumption"] - result["predicted_demand"]
        production_deviation = result["actual_production"] - result["recommended_production"]
        st.markdown("### End-of-Day Outcome")
        st.caption("Compare today's forecast, recommendation, production, and actual consumption.")

        st.markdown("#### Today's numbers")
        st.markdown(
            f"""
            <div class="eod-breakdown">
                <div class="eod-stat" style="--eod-accent:#4d9fff;">
                    <div class="eod-stat-label">Predicted demand</div>
                    <div class="eod-stat-value">{result['predicted_demand']:,} meals</div>
                    <div class="eod-stat-note">Forecast from the demand model</div>
                </div>
                <div class="eod-stat" style="--eod-accent:#50c878;">
                    <div class="eod-stat-label">Recommended production</div>
                    <div class="eod-stat-value">{result['recommended_production']:,} meals</div>
                    <div class="eod-stat-note">Predicted demand plus {result['safety_buffer']:.1f}% safety buffer</div>
                </div>
                <div class="eod-stat" style="--eod-accent:#e6ad45;">
                    <div class="eod-stat-label">Actual production</div>
                    <div class="eod-stat-value">{result['actual_production']:,} meals</div>
                    <div class="eod-stat-note">Meals prepared by the kitchen</div>
                </div>
                <div class="eod-stat" style="--eod-accent:#70d49b;">
                    <div class="eod-stat-label">Actual consumption</div>
                    <div class="eod-stat-value">{result['actual_consumption']:,} meals</div>
                    <div class="eod-stat-note">Meals served and consumed</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### How today performed")
        surplus_accent = "#70d49b" if actual_surplus == 0 else "#e6ad45"
        surplus_status = "No surplus" if actual_surplus == 0 else f"{actual_surplus:,} meals remaining"
        with st.container(border=True):
            st.markdown('<div class="eod-result-label">Actual surplus</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="eod-result-value">{actual_surplus:,} meals</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="status-chip" style="color:{surplus_accent};">{surplus_status}</span>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="eod-block-heading">Calculation</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="eod-formula">Actual production − Actual consumption'
                f'<span class="eod-formula-note">{result["actual_production"]:,} − {result["actual_consumption"]:,} = {actual_surplus:,} meals</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="eod-block-heading">Meaning</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="eod-meaning">{end_of_day_surplus_reading(actual_surplus)}</div>',
                unsafe_allow_html=True,
            )
            if actual_surplus > 0:
                st.markdown(
                    '<div class="eod-note">These meals remain available as surplus and can be reviewed for '
                    'redistribution from the Recipient Matching section.</div>',
                    unsafe_allow_html=True,
                )
        demand_accent = "#70d49b" if prediction_deviation == 0 else ("#e6ad45" if prediction_deviation > 0 else "#4d9fff")
        production_accent = "#70d49b" if production_deviation == 0 else ("#e6ad45" if production_deviation > 0 else "#4d9fff")
        variance_display = f"{prediction_deviation:+,}" if prediction_deviation != 0 else "0"
        production_display = f"{production_deviation:+,}" if production_deviation != 0 else "0"

        with st.container(border=True):
            st.markdown('<div class="eod-result-label">Demand prediction variance</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="eod-result-value">{variance_display} meals</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="status-chip" style="color:{demand_accent};">'
                f'{end_of_day_variance_status(prediction_deviation, "prediction")}</span>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="eod-block-heading">Calculation</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="eod-formula">Actual consumption − Predicted demand'
                f'<span class="eod-formula-note">{result["actual_consumption"]:,} − {result["predicted_demand"]:,} = {variance_display} meals</span>'
                f'<span class="eod-formula-note">{end_of_day_variance_note(prediction_deviation, "prediction")}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="eod-block-heading">Meaning</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="eod-meaning">{end_of_day_demand_reading(prediction_deviation)}</div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown('<div class="eod-result-label">Actual vs recommended production</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="eod-result-value">{production_display} meals</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="status-chip" style="color:{production_accent};">'
                f'{end_of_day_variance_status(production_deviation, "recommendation")}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="eod-pair">Recommended production <strong>'
                f'{result["recommended_production"]:,} meals</strong> · Actual production '
                f'<strong>{result["actual_production"]:,} meals</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="eod-block-heading">Calculation</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="eod-formula">Actual production − Recommended production'
                f'<span class="eod-formula-note">{result["actual_production"]:,} − {result["recommended_production"]:,} = {production_display} meals</span>'
                f'<span class="eod-formula-note">{end_of_day_variance_note(production_deviation, "recommendation")}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="eod-block-heading">Meaning</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="eod-meaning">{end_of_day_production_reading(production_deviation)}</div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown('<div class="eod-result-label">Today\'s interpretation</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="eod-interpretation">'
                f'{end_of_day_demand_reading(prediction_deviation)} '
                f'{end_of_day_surplus_reading(actual_surplus)} '
                f'{end_of_day_production_reading(production_deviation)}</div>',
                unsafe_allow_html=True,
            )

        if st.button("Save Today's Results", type="primary", key="save_end_of_day_result"):
            item_rows = result.get("item_rows") or []
            result_date = pd.Timestamp(result["date"]).date()
            validation_error = None
            if production_breakdown is None or not selected_menu_items or not item_rows:
                validation_error = "No menu items are selected for today, so there is nothing to save."
            else:
                invalid_items = [
                    row["menu_item"] for row in item_rows
                    if int(row["actual_consumption"]) > int(row["actual_production"])
                ]
                if invalid_items:
                    validation_error = (
                        "Actual consumption cannot exceed actual production for: "
                        + ", ".join(invalid_items)
                        + "."
                    )
            saved_daily_dates = (
                set(pd.to_datetime(history["date"], errors="coerce").dropna().dt.date)
                if not history.empty else set()
            )
            saved_item_dates = (
                set(pd.to_datetime(item_history["date"], errors="coerce").dropna().dt.date)
                if not item_history.empty else set()
            )
            if validation_error is None and (
                result_date in saved_daily_dates or result_date in saved_item_dates
            ):
                validation_error = (
                    f"Results for {result_date.isoformat()} have already been saved. "
                    "Completed days cannot be overwritten."
                )
            if validation_error is not None:
                st.error(validation_error)
            else:
                item_records = pd.DataFrame(
                    [
                        {
                            "date": result_date.isoformat(),
                            "menu_item": row["menu_item"],
                            "predicted_quantity": int(row["predicted_quantity"]),
                            "actual_production": int(row["actual_production"]),
                            "actual_consumption": int(row["actual_consumption"]),
                            "surplus_quantity": (
                                int(row["actual_production"]) - int(row["actual_consumption"])
                            ),
                        }
                        for row in item_rows
                    ]
                )
                daily_record = pd.DataFrame(
                    [
                        {
                            "date": result_date.isoformat(),
                            "predicted_demand": result["predicted_demand"],
                            "actual_production": result["actual_production"],
                            "actual_consumption": result["actual_consumption"],
                            "surplus_quantity": actual_surplus,
                            "holiday": result["holiday"],
                            "special_event": result["special_event"],
                            "weather": result["weather"],
                            "precipitation": result["precipitation"],
                        }
                    ]
                )
                inventory_batches = create_inventory_batches(
                    result_date,
                    {
                        row["menu_item"]: int(row["actual_production"]) - int(row["actual_consumption"])
                        for row in item_rows
                    },
                    shelf_life,
                    inventory,
                )
                try:
                    append_item_history(DATA_DIR / "item_history.csv", item_records)
                    append_daily_history(DATA_DIR / "daily_history.csv", daily_record)
                    if not inventory_batches.empty:
                        append_inventory(DATA_DIR / "inventory.csv", inventory_batches)
                except ValueError as error:
                    st.error(str(error))
                else:
                    saved_message = (
                        f"Results for {result_date.isoformat()} were saved to "
                        "daily_history.csv and item_history.csv."
                    )
                    if not inventory_batches.empty:
                        saved_message += (
                            f" {len(inventory_batches)} surplus batch(es) added to Surplus Inventory."
                        )
                    # Confirmed actual production consumes raw ingredients (FEFO).
                    # Keyed on the production event, so a rerun cannot deduct twice.
                    try:
                        raw_outcome = deduct_for_production(
                            RAW_INVENTORY_PATH,
                            RAW_USAGE_PATH,
                            result_date,
                            {
                                row["menu_item"]: int(row["actual_production"])
                                for row in item_rows
                            },
                        )
                        if raw_outcome["already_recorded"]:
                            saved_message += (
                                " Raw ingredients were already deducted for this "
                                "production event, so nothing was deducted again."
                            )
                        elif raw_outcome["deducted"]:
                            saved_message += (
                                f" Raw ingredients deducted FEFO across "
                                f"{len({d['raw_id'] for d in raw_outcome['deducted']})} batch(es)."
                            )
                        if raw_outcome["shortfalls"]:
                            saved_message += (
                                " Insufficient raw stock for: "
                                + ", ".join(raw_outcome["shortfalls"])
                                + " (only available stock was consumed)."
                            )
                        if raw_outcome["skipped"]:
                            saved_message += (
                                " Skipped (no recipe): " + "; ".join(raw_outcome["skipped"])
                            )
                    except (RawInventoryError, KeyError, TypeError, ValueError) as raw_error:
                        # End-of-day results stay saved even if raw stock cannot update.
                        st.warning(
                            f"Results were saved, but raw inventory could not be updated: {raw_error}"
                        )
                    st.session_state["end_of_day_saved_message"] = saved_message
                    st.session_state.pop("end_of_day_result", None)
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()

    st.markdown("### Historical End-of-Day Results")
    completed_dates = pd.to_datetime(history["date"], errors="coerce").dropna() if not history.empty else pd.Series(dtype="datetime64[ns]")
    if completed_dates.empty:
        st.info("No completed end-of-day results are available yet.")
    else:
        eod_from_col, eod_to_col = st.columns(2)
        with eod_from_col:
            eod_start = st.date_input("From", value=completed_dates.min().date(), key="eod_history_from")
        with eod_to_col:
            eod_end = st.date_input("To", value=completed_dates.max().date(), key="eod_history_to")
        if eod_start > eod_end:
            st.error("The From date must be on or before the To date.")
        else:
            completed_history = history.copy()
            completed_history["date"] = pd.to_datetime(completed_history["date"], errors="coerce")
            completed_history = completed_history[
                (completed_history["date"] >= pd.Timestamp(eod_start))
                & (completed_history["date"] <= pd.Timestamp(eod_end))
            ].copy()
            if completed_history.empty:
                st.info("No completed outcomes are available for this period.")
            else:
                completed_history["Prediction Error"] = (
                    completed_history["actual_consumption"] - completed_history["predicted_demand"]
                ).abs()
                st.dataframe(
                    completed_history[
                        [
                            "date", "predicted_demand", "actual_production",
                            "actual_consumption", "surplus_quantity", "Prediction Error",
                        ]
                    ].rename(
                        columns={
                            "date": "Date",
                            "predicted_demand": "Predicted Demand",
                            "actual_production": "Actual Production",
                            "actual_consumption": "Actual Consumption",
                            "surplus_quantity": "Actual Surplus",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )

elif active_section == "Surplus Inventory":
    st.subheader("Surplus Inventory")
    st.caption("Surplus batches from End of Day that have not yet been redistributed.")
    if inventory.empty:
        st.info(
            "No surplus inventory yet. Positive item surplus recorded in End of the Day "
            "will appear here automatically."
        )
    else:
        today = pd.Timestamp.today().date()
        display_rows = []
        for _, inventory_row in inventory.iterrows():
            row_status = calculate_status(
                inventory_row["remaining_quantity"], inventory_row["expiry_date"], today
            )
            display_rows.append(
                {
                    "Menu Item": inventory_row["menu_item"],
                    "Original Quantity": int(inventory_row["original_quantity"]),
                    "Remaining Quantity": int(inventory_row["remaining_quantity"]),
                    "Date Added": pd.Timestamp(inventory_row["date_added"]).strftime("%d %b %Y"),
                    "Expiry Date": pd.Timestamp(inventory_row["expiry_date"]).strftime("%d %b %Y"),
                    "Days Left": days_left_text(inventory_row["expiry_date"], today),
                    "Status": row_status,
                    "_expiry": pd.Timestamp(inventory_row["expiry_date"]),
                    "_inventory_id": inventory_row["inventory_id"],
                }
            )
        display_rows.sort(key=lambda row: (row["_expiry"], row["Menu Item"]))

        status_counts = {"Available": 0, "Expiring Soon": 0, "Expired": 0, "Depleted": 0}
        for row in display_rows:
            status_counts[row["Status"]] += 1
        total_remaining = sum(row["Remaining Quantity"] for row in display_rows)

        with st.container(border=True):
            summary_columns = st.columns(4)
            summary_columns[0].metric("Total remaining surplus", f"{total_remaining:,}")
            summary_columns[1].metric("Expiring soon", status_counts["Expiring Soon"])
            summary_columns[2].metric("Expired", status_counts["Expired"])
            summary_columns[3].metric("Depleted", status_counts["Depleted"])

        active_rows = [row for row in display_rows if row["Remaining Quantity"] > 0]
        inventory_view = pd.DataFrame(
            [
                {
                    "Menu Item": row["Menu Item"],
                    "Original Quantity": row["Original Quantity"],
                    "Remaining Quantity": row["Remaining Quantity"],
                    "Date Added": row["Date Added"],
                    "Expiry Date": row["Expiry Date"],
                    "Days Left": row["Days Left"],
                    "Status": row["Status"],
                }
                for row in active_rows
            ]
        )
        status_colors = {
            "Available": "#70d49b",
            "Expiring Soon": "#ef6f6c",
            "Expired": "#7f8b9c",
            "Depleted": "#aab7c8",
        }
        inventory_style = inventory_view.style.apply(
            lambda column: [
                f"color: {status_colors.get(value, '#9eacbf')}; font-weight: 700;"
                for value in column
            ],
            subset=["Status"],
        )
        st.dataframe(inventory_style, hide_index=True, width="stretch")
        st.caption(
            "Expiry uses the demo shelf-life configuration in data/surplus_shelf_life.csv "
            "(prototype values only, not food-safety guidance)."
        )

        redistributable = [row for row in active_rows if row["Remaining Quantity"] > 0]
        if redistributable:
            st.button(
                "Redistribute All",
                key="redistribute_all",
                type="secondary",
                width="stretch",
                on_click=start_redistribute_all,
                args=(redistributable,),
            )
            st.markdown("### Redistribute surplus")
            for row in redistributable:
                with st.container(border=True):
                    item_col, remaining_col, action_col = st.columns([2, 2, 1])
                    item_col.markdown(f"**{row['Menu Item']}**")
                    remaining_col.markdown(
                        f"{row['Remaining Quantity']:,} remaining · expires {row['Expiry Date']}"
                    )
                    action_col.button(
                        "Redistribute",
                        key=f"redistribute_{row['_inventory_id']}",
                        width="stretch",
                        on_click=start_inventory_redistribution,
                        args=(row["_inventory_id"], row["Menu Item"], row["Remaining Quantity"]),
                    )

elif active_section == "Settings":
    st.subheader("Settings")
    st.caption("Configuration for this kitchen workspace. Changes apply to the existing planning and expiry logic.")

    st.markdown("### Production planning")
    st.caption("Controls the Baseline Preparedness calculation in Raw Inventory.")
    with st.form("settings_planning_form"):
        plan_cols = st.columns(2)
        new_baseline_window = plan_cols[0].number_input(
            "Baseline window (days)", min_value=1,
            value=int(kitchen_settings["baseline_window_days"]), step=1,
        )
        new_safety_buffer = plan_cols[1].number_input(
            "Safety buffer (%)", min_value=0.0,
            value=float(kitchen_settings["safety_buffer_pct"]), step=1.0, format="%.1f",
        )
        st.caption("Baseline window: number of completed days used to calculate the recommended production baseline.")
        st.caption("Safety buffer: extra capacity added above recent average demand.")
        plan_submitted = st.form_submit_button("Save planning settings")
    if plan_submitted:
        try:
            save_settings(SETTINGS_PATH, {
                **kitchen_settings,
                "baseline_window_days": new_baseline_window,
                "safety_buffer_pct": new_safety_buffer,
            })
            st.cache_data.clear()
            st.success("Planning settings saved. Baseline Preparedness has been updated.")
            st.rerun()
        except SettingsError as settings_error:
            st.error(f"Settings not saved: {settings_error}")

    st.markdown("### Inventory & expiry")
    st.caption("Controls Raw Inventory expiry status and the Expiry Watch notifications.")
    with st.form("settings_expiry_form"):
        expiry_cols = st.columns(2)
        new_expiring_soon = expiry_cols[0].number_input(
            "Expiring soon (days)", min_value=0,
            value=int(kitchen_settings["expiring_soon_days"]), step=1,
        )
        new_approaching = expiry_cols[1].number_input(
            "Approaching expiry (days)", min_value=0,
            value=int(kitchen_settings["approaching_expiry_days"]), step=1,
        )
        st.caption("Expiring soon: days remaining at which a batch is flagged Expiring Soon.")
        st.caption("Approaching expiry: days remaining at which a batch is flagged as nearing expiry.")
        expiry_submitted = st.form_submit_button("Save expiry settings")
    if expiry_submitted:
        try:
            save_settings(SETTINGS_PATH, {
                **kitchen_settings,
                "expiring_soon_days": new_expiring_soon,
                "approaching_expiry_days": new_approaching,
            })
            st.cache_data.clear()
            st.success("Expiry settings saved. Raw Inventory status and notifications have been updated.")
            st.rerun()
        except SettingsError as settings_error:
            st.error(f"Settings not saved: {settings_error}")

    st.markdown("### Notifications")
    st.caption("Choose which alerts appear in the Raw Inventory Notification Center. In-app only.")
    with st.form("settings_notify_form"):
        notify_stock = st.checkbox(
            "Stock / unmapped ingredient alerts",
            value=bool(kitchen_settings.get("stock_alerts", True)),
        )
        notify_expiry = st.checkbox(
            "Expiry alerts", value=bool(kitchen_settings.get("expiry_alerts", True))
        )
        notify_prediction = st.checkbox(
            "Today's prediction feasibility alerts",
            value=bool(kitchen_settings.get("prediction_alerts", True)),
        )
        notify_submitted = st.form_submit_button("Save notification preferences")
    if notify_submitted:
        try:
            save_settings(SETTINGS_PATH, {
                **kitchen_settings,
                "stock_alerts": notify_stock,
                "expiry_alerts": notify_expiry,
                "prediction_alerts": notify_prediction,
            })
            st.cache_data.clear()
            st.success("Notification preferences saved.")
            st.rerun()
        except SettingsError as settings_error:
            st.error(f"Settings not saved: {settings_error}")

    st.markdown("### Surplus shelf life")
    st.caption(
        "Set how many days each prepared menu item remains usable after being added "
        "to Surplus Inventory."
    )
    shelf_overrides = kitchen_settings.get(SURPLUS_SHELF_LIFE_KEY) or {}
    with st.form("settings_shelf_life_form"):
        shelf_inputs = {}
        for menu_item in MENU_ITEMS:
            # Show the effective value: the override if set, otherwise the CSV value.
            current = shelf_overrides.get(menu_item, shelf_life.get(menu_item, 1))
            shelf_inputs[menu_item] = st.number_input(
                f"{menu_item} (days)", min_value=1, value=int(current), step=1,
                key=f"shelf_life_{menu_item}",
            )
        st.caption(
            "Expiry is calculated as the date the surplus batch is added plus these days. "
            "Changing a value updates active surplus batches and applies to newly created batches."
        )
        shelf_submitted = st.form_submit_button("Save surplus shelf life")
    if shelf_submitted:
        try:
            save_settings(SETTINGS_PATH, {
                **kitchen_settings,
                SURPLUS_SHELF_LIFE_KEY: dict(shelf_inputs),
            })
            update_active_surplus_shelf_life(
                DATA_DIR / "inventory.csv",
                dict(shelf_inputs),
                today=pd.Timestamp(operation_date).date(),
            )
            st.cache_data.clear()
            st.success("Surplus shelf life saved. Active and new surplus batches will use these durations.")
            st.rerun()
        except SettingsError as settings_error:
            st.error(f"Settings not saved: {settings_error}")

    st.markdown("### System information")
    st.dataframe(
        pd.DataFrame({
            "Item": [
                "Application", "Version", "Menu items", "Raw ingredients",
                "Demand model", "FEFO", "Recipe / BOM", "Raw usage history",
            ],
            "Value": [
                "SMAR System", "SIH 2026 Prototype", str(len(MENU_ITEMS)),
                str(len(RAW_INGREDIENTS)), "Active", "Enabled", "Enabled", "Enabled",
            ],
        }),
        hide_index=True,
        width="stretch",
    )

    st.markdown("### Restore defaults")
    st.warning(
        "This resets configuration only. Inventory, usage history, production, "
        "surplus and redistribution records are not affected."
    )
    confirm_restore = st.checkbox("Yes, restore default settings", key="settings_restore_confirm")
    if st.button("Restore Defaults", disabled=not confirm_restore, key="settings_restore_btn"):
        restore_defaults(SETTINGS_PATH)
        st.cache_data.clear()
        st.success("Default settings restored.")
        st.rerun()
