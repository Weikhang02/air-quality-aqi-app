import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="7-Day AQI Forecasting Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL + FEATURES + TARGETS
# -----------------------------
@st.cache_resource
def load_models():
    model = joblib.load("rf_aqi_8output_model.pkl")
    feature_cols = joblib.load("aqi_feature_cols.pkl")
    target_cols = joblib.load("aqi_target_cols.pkl")
    return model, feature_cols, target_cols

try:
    model, feature_cols, target_cols = load_models()
except:
    st.error("⚠️ Model files not found. Please ensure the model files are in the same directory.")
    st.stop()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_aqi_category(aqi):
    """Return AQI category and color"""
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"

def create_gauge_chart(aqi_value, title):
    """Create a gauge chart for AQI"""
    category, color = get_aqi_category(aqi_value)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = aqi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#90EE90'}},
        number = {'font': {'color': '#90EE90'}},
        gauge = {
            'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "#90EE90", 'tickfont': {'color': '#90EE90'}},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#00e400'},
                {'range': [50, 100], 'color': '#ffff00'},
                {'range': [100, 150], 'color': '#ff7e00'},
                {'range': [150, 200], 'color': '#ff0000'},
                {'range': [200, 300], 'color': '#8f3f97'},
                {'range': [300, 500], 'color': '#7e0023'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "white", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_forecast_chart(result_df):
    """Create line chart for 7-day forecast"""
    fig = go.Figure()
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=result_df['Day'],
        y=result_df['Predicted_AQI'],
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=10, color='#2E7D32')
    ))
    
    # Add AQI bands
    fig.add_hrect(y0=0, y1=50, fillcolor="#00e400", opacity=0.1, line_width=0, annotation_text="Good", annotation_position="right")
    fig.add_hrect(y0=50, y1=100, fillcolor="#ffff00", opacity=0.1, line_width=0, annotation_text="Moderate", annotation_position="right")
    fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e00", opacity=0.1, line_width=0, annotation_text="Unhealthy (Sensitive)", annotation_position="right")
    fig.add_hrect(y0=150, y1=200, fillcolor="#ff0000", opacity=0.1, line_width=0, annotation_text="Unhealthy", annotation_position="right")
    
    fig.update_layout(
        title="7-Day AQI Forecast",
        xaxis_title="Day",
        yaxis_title="AQI Value",
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# -----------------------------
# HEADER
# -----------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌍 AQI Forecasting Dashboard")
    st.markdown("**Predict current AQI and forecast for the next 7 days**")
with col2:
    st.image("aqi.png", width=150)

st.markdown("---")

# -----------------------------
# SIDEBAR USER INPUT
# -----------------------------
st.sidebar.title("📊 Input Parameters")

# Initialize user_input
user_input = {}

# Choice between sliders or number inputs
input_type = st.sidebar.selectbox(
    "Input type:",
    ["Sliders", "Number Input"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Primary Pollutants")

for feat in feature_cols:
    if input_type == "Sliders":
        user_input[feat] = st.sidebar.slider(
            f"{feat}",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            help=f"Adjust {feat} concentration"
        )
    else:
        user_input[feat] = st.sidebar.number_input(
            f"{feat}",
            value=0.0,
            format="%.3f",
            help=f"Enter the current value for {feat}"
        )

# -----------------------------
# RESET BUTTON
# -----------------------------
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset All Values"):
    st.rerun()

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Predict AQI", use_container_width=True)

# -----------------------------
# INFO SECTION
# -----------------------------
with st.expander("ℹ️ About AQI Categories"):
    info_df = pd.DataFrame({
        'AQI Range': ['0-50', '51-100', '101-150', '151-200', '201-300', '301-500'],
        'Category': ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
        'Health Implications': [
            'Air quality is satisfactory',
            'Acceptable for most people',
            'Members of sensitive groups may experience effects',
            'Everyone may begin to experience effects',
            'Health alert: everyone may experience serious effects',
            'Health warning of emergency conditions'
        ]
    })
    st.table(info_df)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
if predict_button:
    with st.spinner('🔄 Calculating predictions...'):
        # Convert to DataFrame
        X_new = pd.DataFrame([user_input], columns=feature_cols)
        
        # Predict
        preds = model.predict(X_new)[0]
        
        # Prepare results
        result_df = pd.DataFrame({
            "Day": [
                "Today",
                "Day +1",
                "Day +2",
                "Day +3",
                "Day +4",
                "Day +5",
                "Day +6",
                "Day +7"
            ],
            "Predicted_AQI": preds
        })
        
        # Round AQI values
        result_df['Predicted_AQI'] = result_df['Predicted_AQI'].round(1)
        
        st.success("✅ Prediction completed!")
        
        # Today's AQI with gauge
        st.subheader("📍 Current AQI")
        col1, col2 = st.columns([1, 2])
        
        today_aqi = preds[0]
        category, color = get_aqi_category(today_aqi)
        
        with col1:
            st.plotly_chart(create_gauge_chart(today_aqi, "Today's AQI"), use_container_width=True)
        
        with col2:
            st.markdown(f"### AQI: {today_aqi:.1f}")
            st.markdown(f"**Category:** <span style='color:{color}; font-size:24px; font-weight:bold'>{category}</span>", unsafe_allow_html=True)
            
            # Health recommendations
            if today_aqi <= 50:
                st.info("🟢 Air quality is good. Enjoy outdoor activities!")
            elif today_aqi <= 100:
                st.warning("🟡 Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.")
            elif today_aqi <= 150:
                st.warning("🟠 Sensitive groups should reduce prolonged outdoor exertion.")
            elif today_aqi <= 200:
                st.error("🔴 Everyone should reduce prolonged outdoor exertion.")
            else:
                st.error("🟣 Health alert! Everyone should avoid outdoor exertion.")
        
        st.markdown("---")
        
        # 7-Day Forecast
        st.subheader("📈 7-Day AQI Forecast")
        
        # Chart
        st.plotly_chart(create_forecast_chart(result_df), use_container_width=True)
        
        # Table with categories
        result_df['Category'] = result_df['Predicted_AQI'].apply(lambda x: get_aqi_category(x)[0])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_aqi = result_df['Predicted_AQI'].mean()
            st.metric("Average AQI", f"{avg_aqi:.1f}", delta=None)
        
        with col2:
            max_aqi = result_df['Predicted_AQI'].max()
            max_day = result_df.loc[result_df['Predicted_AQI'].idxmax(), 'Day']
            st.metric("Peak AQI", f"{max_aqi:.1f}", delta=f"on {max_day}")
        
        with col3:
            min_aqi = result_df['Predicted_AQI'].min()
            min_day = result_df.loc[result_df['Predicted_AQI'].idxmin(), 'Day']
            st.metric("Best AQI", f"{min_aqi:.1f}", delta=f"on {min_day}")
        
        with col4:
            trend = "📈 Rising" if preds[-1] > preds[0] else "📉 Falling"
            st.metric("Trend", trend)
        
        # Detailed table
        with st.expander("📋 View Detailed Data"):
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="aqi_forecast.csv",
                mime="text/csv"
            )

else:
    # Welcome message
    st.info("👈 Enter pollutant values in the sidebar and click 'Predict AQI' to get started!")
    
    # Sample visualization
    st.subheader("🎯 How it works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Input Data")
        st.write("Enter current pollutant concentrations in the sidebar")
    
    with col2:
        st.markdown("### 2️⃣ AI Prediction")
        st.write("Our model analyzes patterns and forecasts AQI")
    
    with col3:
        st.markdown("### 3️⃣ Visualize Results")
        st.write("View current AQI and 7-day forecast with insights")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit | Data updated in real-time</p>
    </div>
    """,
    unsafe_allow_html=True
)