import pandas as pd
import pydeck as pdk


KITCHEN_LOCATION = {
    "name": "Main Kitchen",
    "latitude": 34.0835,
    "longitude": 74.7970,
}

PRIORITY_COLORS = {
    "High": [220, 53, 69],
    "Medium": [255, 193, 7],
    "Low": [25, 135, 84],
}


def build_route_map(redistribution_plan):
    """
    Build an interactive PyDeck map showing kitchen, recipients, and route.
    
    Args:
        redistribution_plan: DataFrame with recipient locations and allocations
        
    Returns:
        pdk.Deck: Interactive map object ready for st.pydeck_chart()
    """
    if redistribution_plan.empty:
        return None

    recipient_points = redistribution_plan.copy()

    recipient_points["fill_color"] = recipient_points["Priority"].map(
        lambda p: PRIORITY_COLORS.get(p, [108, 117, 125])
    )

    kitchen_data = pd.DataFrame(
        [
            {
                "name": KITCHEN_LOCATION["name"],
                "latitude": KITCHEN_LOCATION["latitude"],
                "longitude": KITCHEN_LOCATION["longitude"],
                "fill_color": [13, 110, 253],
                "type": "kitchen",
            }
        ]
    )

    route_path = [
        [KITCHEN_LOCATION["longitude"], KITCHEN_LOCATION["latitude"]],
        *[
            [row["longitude"], row["latitude"]]
            for _, row in recipient_points.iterrows()
        ],
    ]

    layers = [
        pdk.Layer(
            "PathLayer",
            data=[{"path": route_path}],
            get_path="path",
            get_color=[22, 121, 77],
            width_min_pixels=4,
            width_max_pixels=8,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=recipient_points,
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_radius="Meals allocated * 5 + 50",
            pickable=True,
            highlight_color=[255, 0, 0],
            get_line_color=[0, 0, 0],
            line_width_min_pixels=2,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=kitchen_data,
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_radius=150,
            pickable=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=3,
        ),
    ]

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=KITCHEN_LOCATION["latitude"],
            longitude=KITCHEN_LOCATION["longitude"],
            zoom=12,
            pitch=0,
        ),
        tooltip={
            "html": (
                "<b>{Recipient}</b><br/>"
                "Meals: {Meals allocated}<br/>"
                "Priority: {Priority}<br/>"
                "Distance: {Distance (km):.1f} km"
            ),
            "style": {"backgroundColor": "#0f5132", "color": "white", "fontSize": "12px"},
        },
    )


def get_recipient_markers(recipients):
    """
    Convert recipient data to map markers format.
    
    Args:
        recipients: DataFrame with recipient locations
        
    Returns:
        DataFrame suitable for st.map()
    """
    map_data = recipients[["name", "latitude", "longitude", "priority", "capacity"]].copy()
    map_data = map_data.rename(columns={"latitude": "lat", "longitude": "lon"})
    return map_data
