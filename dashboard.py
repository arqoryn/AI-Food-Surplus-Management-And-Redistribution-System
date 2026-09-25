import math
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_preprocessing import (
    DAILY_COLUMNS,
    HISTORY_COLUMNS,
    RECIPIENT_COLUMNS,
    RECORD_COLUMNS,
    load_csv,
    save_redistribution_records,
    validate_columns,
)
from src.demand_prediction import predict_demand, train_or_load_model


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "demand_model.joblib"


@st.cache_data
def load_operational_data():
    history = load_csv(DATA_DIR / "kitchen_history.csv", HISTORY_COLUMNS)
    daily = load_csv(DATA_DIR / "daily_operations.csv", DAILY_COLUMNS)
    recipients = load_csv(DATA_DIR / "recipients.csv", RECIPIENT_COLUMNS)
    records = load_csv(DATA_DIR / "redistribution_records.csv", RECORD_COLUMNS)
    return history, daily, recipients, records


@st.cache_resource
def get_model(history):
    return train_or_load_model(history, MODEL_PATH)


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
            }
        )
        remaining -= allocated
        if remaining == 0:
            break
    return pd.DataFrame(plan), remaining


def append_plan_records(plan, operation_date):
    records = pd.DataFrame(
        {
            "date": operation_date,
            "recipient_id": plan["recipient_id"],
            "allocated_quantity": plan["Meals allocated"],
            "distance_km": plan["Distance (km)"],
            "status": "Planned",
        }
    )
    save_redistribution_records(DATA_DIR / "redistribution_records.csv", records)


st.set_page_config(page_title="FoodLoop AI", page_icon=":material/compost:", layout="wide")
st.markdown(
    """
    <style>
    .hero {
        padding: 1.3rem 1.5rem;
        border-radius: 1rem;
        background: linear-gradient(120deg, #0f5132 0%, #16794d 65%, #2d9b68 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; color: white; font-size: 2.2rem; }
    .hero p { margin: .35rem 0 0; color: #e5f5eb; font-size: 1rem; }
    .section-label {
        color: #16794d;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    </style>
    <div class="hero">
        <h1>FoodLoop AI</h1>
        <p>Smarter production decisions. More meals reaching people.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    history, daily_operations, recipients, records = load_operational_data()
    model = get_model(history)
except (FileNotFoundError, ValueError) as error:
    st.error(f"Data configuration error: {error}")
    st.stop()

default_operation = daily_operations.iloc[-1]
with st.sidebar:
    st.header("Kitchen scenario")
    with st.form("scenario_form"):
        operation_date = st.date_input("Operation date", value=pd.to_datetime(default_operation["date"]).date())
        meal_type = st.selectbox("Meal type", sorted(history["meal_type"].unique()), index=0)
        menu_type = st.selectbox("Menu type", sorted(history["menu_type"].unique()), index=0)
        planned_production = st.number_input(
            "Planned production (meals)", min_value=0, value=int(default_operation["planned_quantity"]), step=10
        )
        holiday = st.checkbox("Holiday", value=bool(default_operation["holiday"]))
        special_event = st.checkbox("Special event or high-footfall day", value=bool(default_operation["special_event"]))
        temperature = st.number_input(
            "Temperature (°C)", min_value=-10.0, max_value=50.0,
            value=float(history["temperature"].mean()), step=0.1, format="%.1f",
        )
        rainfall = st.number_input(
            "Rainfall (mm)", min_value=0.0, max_value=500.0,
            value=float(history["rainfall"].mean()), step=0.1, format="%.1f",
        )
        st.form_submit_button("Update forecast", type="primary", icon=":material/refresh:")

    st.divider()
    st.caption("Data sources")
    st.write(f"Historical demand: {len(history)} records")
    st.write(f"Active recipients: {int(recipients['active'].sum())}")
    st.write(f"Last delivery: {records['date'].max()}")

prediction_input = pd.DataFrame(
    [
        {
            "date": operation_date,
            "meal_type": meal_type,
            "menu_type": menu_type,
            "holiday": int(holiday),
            "special_event": int(special_event),
            "temperature": temperature,
            "rainfall": rainfall,
        }
    ]
)
predicted_demand = predict_demand(model, prediction_input)
expected_surplus = max(int(planned_production) - predicted_demand, 0)
waste_risk = min(expected_surplus / max(int(planned_production), 1) * 100, 100)
recommended_production = math.ceil(predicted_demand * 1.05)
avoidable_surplus = max(int(planned_production) - recommended_production, 0)
redistribution_plan, unallocated_surplus = build_redistribution_plan(expected_surplus, recipients)
redistributed_meals = int(redistribution_plan["Meals allocated"].sum()) if not redistribution_plan.empty else 0
co2_avoided = redistributed_meals * 0.5

with st.container(horizontal=True):
    st.metric("Predicted demand", f"{predicted_demand:,} meals", help="Estimated meals consumed for this scenario.", border=True)
    st.metric("Planned production", f"{int(planned_production):,} meals", help="Meals currently planned by the kitchen.", border=True)
    st.metric("Expected surplus", f"{expected_surplus:,} meals", help="Planned production minus predicted demand.", border=True)
    st.metric("Waste risk", f"{waste_risk:.0f}%", help="Expected surplus as a percentage of planned production.", border=True)

st.markdown('<div class="section-label">Decision centre</div>', unsafe_allow_html=True)
st.subheader("Today at a glance")
status_col, context_col = st.columns([1, 2])
with status_col:
    if expected_surplus == 0:
        st.success("On track: no surplus expected", icon=":material/check_circle:")
    elif unallocated_surplus:
        st.warning("Action needed: recipient capacity is limited", icon=":material/warning:")
    else:
        st.info("Ready: surplus can be redistributed", icon=":material/local_shipping:")
with context_col:
    st.caption(
        f"Scenario for **{pd.Timestamp(operation_date).strftime('%d %b %Y')}** · "
        f"{meal_type} · {menu_type} · "
        f"{'special event' if special_event else 'regular day'}"
    )
overview, redistribution = st.columns(2)
with overview:
    with st.container(border=True):
        st.markdown("**Production recommendation**")
        st.metric("Prepare approximately", f"{recommended_production:,} meals")
        if avoidable_surplus:
            st.info(f"Reducing production to the recommendation could avoid {avoidable_surplus:,} surplus meals.")
        else:
            st.success("Planned production is aligned with the forecast buffer.")
        chart_data = history.assign(date=pd.to_datetime(history["date"])).sort_values("date")
        st.line_chart(chart_data, x="date", y="actual_consumption", x_label="Date", y_label="Meals consumed")

with redistribution:
    with st.container(border=True):
        st.markdown("**Redistribution readiness**")
        st.metric("Meals matched", f"{redistributed_meals:,}", f"{len(redistribution_plan)} recipients")
        if unallocated_surplus:
            st.warning(f"{unallocated_surplus:,} meals still need a recipient with available capacity.")
        elif expected_surplus:
            st.success("All expected surplus is assigned to recipient capacity.")
        else:
            st.info("No surplus is expected for this scenario.")
        st.bar_chart(
            redistribution_plan.set_index("Recipient")[["Meals allocated"]]
            if not redistribution_plan.empty
            else pd.DataFrame({"Meals allocated": []})
        )

forecast_tab, redistribution_tab, impact_tab = st.tabs(
    [":material/insights: Forecast and surplus", ":material/local_shipping: Recipient matching", ":material/monitoring: Impact and analytics"]
)

with forecast_tab:
    st.subheader("Demand forecast")
    chart_data = history.assign(date=pd.to_datetime(history["date"])).sort_values("date")
    forecast_point = pd.DataFrame({"date": [pd.Timestamp(operation_date)], "actual_consumption": [predicted_demand]})
    st.line_chart(pd.concat([chart_data[["date", "actual_consumption"]], forecast_point]).set_index("date"))
    st.dataframe(
        pd.DataFrame(
            {
                "Signal": ["Model inputs", "Holiday adjustment", "Special event adjustment"],
                "Value": [f"{meal_type} / {menu_type}", "Yes" if holiday else "No", "Yes" if special_event else "No"],
            }
        ),
        hide_index=True,
    )

with redistribution_tab:
    st.subheader("Recipient matching and route plan")
    if redistribution_plan.empty:
        st.info("No surplus is available to redistribute in this scenario.")
    else:
        st.dataframe(redistribution_plan.drop(columns=["recipient_id"]), hide_index=True)
        st.caption("Route order prioritizes urgent need first, then balances recipient demand against distance.")
        st.table(
            {
                ":material/route: Route": "Kitchen → " + " → ".join(redistribution_plan["Recipient"]),
                ":material/distance: Total distance": f"{redistribution_plan['Distance (km)'].sum():.1f} km",
                ":material/local_shipping: Meals in transit": f"{redistributed_meals:,}",
            },
            border="horizontal",
            width="content",
        )
        if st.button("Record this redistribution plan", type="primary"):
            append_plan_records(redistribution_plan, operation_date.isoformat())
            st.success("Redistribution plan saved to redistribution_records.csv.")
            st.cache_data.clear()

with impact_tab:
    st.subheader("Impact and analytics")
    delivered = records[records["status"].str.lower() == "delivered"]
    impact_columns = st.columns(4)
    impact_columns[0].metric("Meals redistributed", f"{int(delivered['allocated_quantity'].sum()):,}", border=True)
    impact_columns[1].metric("Recipients served", f"{delivered['recipient_id'].nunique()}", border=True)
    impact_columns[2].metric("Avoided CO₂e", f"{int(delivered['allocated_quantity'].sum()) * 0.5:.1f} kg", border=True)
    impact_columns[3].metric("Recorded deliveries", f"{len(delivered):,}", border=True)
    st.dataframe(recipients, hide_index=True)
    if not delivered.empty:
        st.subheader("Redistribution history")
        st.dataframe(delivered.sort_values("date", ascending=False), hide_index=True)

with st.expander("Historical kitchen data"):
    st.dataframe(history, hide_index=True)
