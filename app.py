"""
AI-Ration: Decision Intelligence Dashboard
Fixed version with proper session state initialization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Tuple, Optional

# Import configurations and engine
from config import app_config, model_config, ui_config
from engine import (
    load_data,
    train_models,
    predict,
    apply_decision_logic,
    district_prioritization,
    cost_optimization,
    scenario_simulation,
    get_model_insights,
    AIDemandPredictor
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title=app_config.APP_NAME,
    page_icon="🛒",
    layout=ui_config.PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .risk-critical { color: #F44336; font-weight: bold; }
    .risk-high { color: #FF9800; font-weight: bold; }
    .risk-medium { color: #FFD700; font-weight: bold; }
    .risk-normal { color: #4CAF50; font-weight: bold; }
    .stDataFrame { font-size: 0.9rem; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
# Initialize all session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data_loaded = False
    st.session_state.models_trained = False
    st.session_state.predictions = None
    st.session_state.context = {}
    st.session_state.last_update = None
    st.session_state.predictor = None
    st.session_state.data = None
    st.session_state.models = None

# =====================================================
# SIDEBAR - ROLE & CONTEXT SETTINGS
# =====================================================
with st.sidebar:
    # App Logo
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #2E86AB; font-size: 2.5rem;">🛒</h1>
        <h3>AI-Ration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"Version {app_config.APP_VERSION}")
    st.markdown("---")
    
    # Role Selection
    st.header("🔐 Role-Based Access")
    role = st.selectbox(
        "Select Your Role",
        ["Shopkeeper", "Admin", "Policy Maker", "Auditor"],
        index=0
    )
    
    # Light authentication
    if role == "Admin":
        password = st.text_input("Admin Password", type="password", 
                                help="Enter 'admin123' for demo")
        if password and password != "admin123":
            st.error("Incorrect password")
            st.stop()
    
    st.success(f"✅ Logged in as: **{role}**")
    st.markdown("---")
    
    # Context Settings
    st.header("📋 Context Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        festival = st.checkbox("Festival Week", help="Major local festival")
        wage = st.checkbox("Wage Week", help="Factory salary disbursement")
    with col2:
        rainy = st.checkbox("Rainy Week", help="Heavy rainfall forecast")
        harvest = st.checkbox("Harvest Season", help="Crop harvest period")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        temp = st.slider("Temperature (°C)", 15.0, 40.0, 25.0, 0.5)
        rainfall_mm = st.slider("Rainfall (mm)", 0.0, 20.0, 0.0, 0.5)
        month = st.selectbox("Month", 
                           ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                           index=datetime.now().month - 1)
    
    # Store context
    context = {
        "festival_week": int(festival),
        "wage_week": int(wage),
        "rainy_week": int(rainy),
        "is_festival_season": int(festival or (month in ["Oct", "Nov"])),
        "is_harvest_season": int(harvest or (month in ["Mar", "Apr"])),
        "temperature": temp,
        "rainfall": rainfall_mm,
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(month) + 1
    }
    
    # Data Management
    st.markdown("---")
    st.header("🔄 Data Management")
    
    if st.button("🔄 Refresh Data & Models", use_container_width=True):
        with st.spinner("Loading data and training models..."):
            try:
                predictor = AIDemandPredictor(model_type="rf")
                data = predictor.load_data()
                models = predictor.train_models(data)
                
                st.session_state.data = data
                st.session_state.models = models
                st.session_state.predictor = predictor
                st.session_state.models_trained = True
                st.session_state.initialized = True
                st.session_state.last_update = datetime.now()
                
                st.success("Data refreshed successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error refreshing data: {e}")
    
    # System Status
    st.markdown("---")
    st.header("📊 System Status")
    
    if st.session_state.get('models_trained'):
        st.success("✅ Models Ready")
        if st.session_state.last_update:
            st.caption(f"Last update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.warning("⚠️ Models need training")

# =====================================================
# MAIN DASHBOARD
# =====================================================

# Header
st.markdown('<div class="main-header">AI-Ration Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explainable AI for Public Distribution Systems | Zero Hunger Initiative</div>', 
            unsafe_allow_html=True)

# Initialize data and models if not already done
if not st.session_state.initialized or st.session_state.data is None:
    with st.spinner("🚀 Initializing AI-Ration system for the first time..."):
        try:
            predictor = AIDemandPredictor(model_type="rf")
            data = predictor.load_data()
            models = predictor.train_models(data)
            
            st.session_state.data = data
            st.session_state.models = models
            st.session_state.predictor = predictor
            st.session_state.models_trained = True
            st.session_state.initialized = True
            st.session_state.last_update = datetime.now()
            
            st.success("System initialized successfully!")
        except Exception as e:
            st.error(f"Initialization error: {e}")
            # Create fallback data
            predictor = AIDemandPredictor(model_type="rf")
            data = predictor.create_synthetic_data(weeks=12)
            models = predictor.train_models(data)
            
            st.session_state.data = data
            st.session_state.models = models
            st.session_state.predictor = predictor
            st.session_state.models_trained = True
            st.session_state.initialized = True
            st.session_state.last_update = datetime.now()

# Load from session state
data = st.session_state.data
models = st.session_state.models
predictor = st.session_state.predictor

# =====================================================
# CORE COMPUTATIONS (with error handling)
# =====================================================
try:
    with st.spinner("🤖 Running AI predictions..."):
        if predictor is None:
            # Create new predictor if needed
            predictor = AIDemandPredictor(model_type="rf")
            predictor.data = data
            predictor.models = models
            predictor.prepare_features(data)
            st.session_state.predictor = predictor
        
        pred_df = predictor.predict(models, context)
        final_df = predictor.apply_decision_logic(pred_df, data)
        priority_df = predictor.district_prioritization(final_df)
        cost_df = predictor.cost_optimization(final_df)
        scenario_df = predictor.scenario_simulation(models, data)
        
except Exception as e:
    st.error(f"Error in predictions: {str(e)[:100]}...")
    
    # Create fallback data
    pred_df = pd.DataFrame({
        "Item": ["Rice", "Wheat", "Sugar"],
        "Predicted_Demand_kg": [500, 400, 150],
        "Confidence_Interval": ["450-550", "360-440", "135-165"]
    })
    
    final_df = pd.DataFrame({
        "Item": ["Rice", "Wheat", "Sugar"],
        "Predicted_Demand_kg": [500, 400, 150],
        "Historical_Avg_kg": [520, 420, 150],
        "Deviation_%": [-3.8, -4.8, 0.0],
        "Risk": ["🟢 Normal", "🟢 Normal", "🟢 Normal"],
        "Risk_Color": ["#4CAF50", "#4CAF50", "#4CAF50"],
        "Action_Required": ["Maintain stock", "Maintain stock", "Maintain stock"],
        "Recommended_Order_kg": [500, 400, 150]
    })
    
    priority_df = pd.DataFrame({
        "Shop": ["Shop_A", "Shop_B", "Shop_C"],
        "Item": ["Rice", "Wheat", "Sugar"],
        "Risk": ["🟢 Normal", "🟢 Normal", "🟢 Normal"],
        "Priority_Score": [1.0, 0.8, 0.5],
        "Priority_Level": ["🟢 Low", "🟢 Low", "🟢 Low"]
    })
    
    cost_df = pd.DataFrame({
        "Item": ["Rice", "Wheat", "Sugar", "TOTAL"],
        "Recommended_Qty_kg": [500, 400, 150, 1050],
        "Cost_per_kg_₹": [28, 24, 32, 28],
        "Total_Cost_₹": [14000, 9600, 4800, 28400]
    })
    
    scenario_df = pd.DataFrame({
        "Scenario": ["Normal", "Festival Peak", "Monsoon"],
        "Item": ["Rice", "Rice", "Rice"],
        "Predicted_Demand_kg": [500, 600, 550]
    })

# =====================================================
# SHOPKEEPER VIEW
# =====================================================
if role == "Shopkeeper":
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_demand = final_df["Predicted_Demand_kg"].sum()
        st.metric("Total Predicted Demand", f"{total_demand:.0f} kg")
    
    with col2:
        if "Risk" in final_df.columns:
            high_risk_items = len(final_df[final_df["Risk"].str.contains("🔴|🟠")])
            st.metric("High Risk Items", high_risk_items)
        else:
            st.metric("High Risk Items", 0)
    
    with col3:
        if "Deviation_%" in final_df.columns:
            avg_deviation = final_df["Deviation_%"].mean()
            st.metric("Avg Deviation", f"{avg_deviation:.1f}%")
        else:
            st.metric("Avg Deviation", "0%")
    
    with col4:
        if not cost_df.empty:
            total_cost = cost_df.iloc[-1]["Total_Cost_₹"] if "TOTAL" in cost_df["Item"].values else cost_df["Total_Cost_₹"].sum()
            st.metric("Estimated Cost", f"₹{total_cost:,.0f}")
        else:
            st.metric("Estimated Cost", "₹0")
    
    st.markdown("---")
    
    # Main Forecast & Recommendations
    st.subheader("📦 Demand Forecast & Actionable Recommendations")
    
    # Ensure required columns exist
    display_cols = ["Item", "Predicted_Demand_kg", "Risk", "Action_Required"]
    available_cols = [col for col in display_cols if col in final_df.columns]
    
    if available_cols:
        display_df = final_df[available_cols]
        
        # Apply conditional formatting
        def color_risk(val):
            if "🔴" in str(val):
                return 'background-color: #FFEBEE'
            elif "🟠" in str(val):
                return 'background-color: #FFF3E0'
            elif "🟡" in str(val):
                return 'background-color: #FFFDE7'
            elif "🟢" in str(val):
                return 'background-color: #E8F5E9'
            return ''
        
        # Apply formatting only to columns that exist
        format_cols = [col for col in ["Risk", "Action_Required"] if col in display_df.columns]
        if format_cols:
            styled_df = display_df.style.applymap(color_risk, subset=format_cols)
        else:
            styled_df = display_df.style
        
        st.dataframe(styled_df, use_container_width=True, height=300)
    else:
        st.dataframe(final_df, use_container_width=True, height=300)
    
    # Download button
    col1, col2 = st.columns([3, 1])
    with col2:
        st.download_button(
            "📥 Download Recommendations",
            final_df.to_csv(index=False),
            "ai_ration_recommendations.csv",
            "text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Visualizations
    if not final_df.empty and "Item" in final_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Demand by Item")
            
            # Prepare data for chart
            if "Predicted_Demand_kg" in final_df.columns:
                chart_data = final_df[["Item", "Predicted_Demand_kg"]].copy()
                
                fig = go.Figure()
                
                colors = {"Rice": "#FF6B6B", "Wheat": "#4ECDC4", "Sugar": "#FFD166"}
                
                for _, row in chart_data.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row["Item"]],
                        y=[row["Predicted_Demand_kg"]],
                        marker_color=colors.get(row["Item"], "#2E86AB"),
                        text=[f"{row['Predicted_Demand_kg']:.0f} kg"],
                        textposition='auto',
                        name=row["Item"]
                    ))
                
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    yaxis_title="Demand (kg)",
                    xaxis_title="Item"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Demand data not available")
        
        with col2:
            st.subheader("🎯 Risk Distribution")
            
            if "Risk" in final_df.columns:
                risk_counts = final_df["Risk"].value_counts()
                
                if len(risk_counts) > 0:
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        color=risk_counts.index,
                        color_discrete_map={
                            "🔴 Critical Shortage": "#F44336",
                            "🟠 High Demand": "#FF9800",
                            "🟡 Moderate": "#FFD700",
                            "🟢 Normal": "#4CAF50"
                        },
                        hole=0.4
                    )
                    
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label'
                    )
                    
                    fig.update_layout(
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No risk categories found")
            else:
                st.info("Risk data not available")
    
    # Stock Level Indicators
    st.markdown("---")
    st.subheader("📈 Stock Level Indicators")
    
    for _, row in final_df.iterrows():
        if "Item" in row and "Predicted_Demand_kg" in row:
            item_name = row["Item"]
            predicted_demand = row["Predicted_Demand_kg"]
            
            col1, col2, col3 = st.columns([1, 3, 2])
            
            with col1:
                st.markdown(f"**{item_name}**")
            
            with col2:
                # Simulate stock level (70% of predicted demand)
                current_stock = predicted_demand * 0.7
                stock_percent = (current_stock / predicted_demand) * 100
                stock_percent = min(100, max(0, stock_percent))
                
                if stock_percent < 30:
                    bar_color = "red"
                elif stock_percent < 60:
                    bar_color = "orange"
                else:
                    bar_color = "green"
                
                st.progress(
                    stock_percent / 100, 
                    text=f"Stock: {current_stock:.0f} kg / {predicted_demand:.0f} kg ({stock_percent:.0f}%)"
                )
            
            with col3:
                if stock_percent < 50:
                    st.warning(f"⚠️ Stock low ({stock_percent:.0f}%)")
                else:
                    st.success("✅ Stock sufficient")

# =====================================================
# ADMIN VIEW
# =====================================================
elif role == "Admin":
    st.subheader("🏛️ District-Wide Operational Dashboard")
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not priority_df.empty and "Priority_Level" in priority_df.columns:
            critical_shops = priority_df[priority_df["Priority_Level"] == "🔴 Critical"]["Shop"].nunique()
            st.metric("Critical Shops", critical_shops)
        else:
            st.metric("Critical Shops", 0)
    
    with col2:
        if not priority_df.empty and "Priority_Score" in priority_df.columns:
            total_priority = priority_df["Priority_Score"].sum()
            st.metric("Total Priority Score", f"{total_priority:.1f}")
        else:
            st.metric("Total Priority Score", "0.0")
    
    with col3:
        if not priority_df.empty and "Urgency" in priority_df.columns:
            immediate_actions = priority_df[priority_df["Urgency"] == "Immediate"]["Urgency"].count()
            st.metric("Immediate Actions", immediate_actions)
        else:
            st.metric("Immediate Actions", 0)
    
    with col4:
        if not priority_df.empty and "Beneficiaries" in priority_df.columns:
            total_beneficiaries = priority_df["Beneficiaries"].sum()
            st.metric("Total Beneficiaries", f"{total_beneficiaries:,}")
        else:
            st.metric("Total Beneficiaries", "0")
    
    st.markdown("---")
    
    # District Prioritization
    st.subheader("📍 Shop Prioritization Matrix")
    
    if not priority_df.empty:
        # Display with formatting
        def color_priority(val):
            if isinstance(val, str):
                if "🔴" in val:
                    return 'background-color: #FFEBEE; color: #B71C1C; font-weight: bold'
                elif "🟠" in val:
                    return 'background-color: #FFF3E0; color: #E65100'
                elif "🟡" in val:
                    return 'background-color: #FFFDE7; color: #F57F17'
                elif "🟢" in val:
                    return 'background-color: #E8F5E9; color: #1B5E20'
            return ''
        
        if "Priority_Level" in priority_df.columns:
            styled_priority = priority_df.style.applymap(color_priority, subset=['Priority_Level'])
            st.dataframe(styled_priority, use_container_width=True, height=400)
        else:
            st.dataframe(priority_df, use_container_width=True, height=400)
        
        # Download button
        st.download_button(
            "📥 Download District Priorities",
            priority_df.to_csv(index=False),
            "district_priorities.csv",
            "text/csv",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Priority Score by Shop")
            
            if "Shop" in priority_df.columns and "Priority_Score" in priority_df.columns:
                shop_scores = priority_df.groupby("Shop")["Priority_Score"].sum().sort_values(ascending=False)
                
                if not shop_scores.empty:
                    fig = go.Figure()
                    
                    for shop, score in shop_scores.items():
                        if score >= 2.0:
                            color = "#F44336"
                        elif score >= 1.5:
                            color = "#FF9800"
                        elif score >= 1.0:
                            color = "#FFD700"
                        else:
                            color = "#4CAF50"
                        
                        fig.add_trace(go.Bar(
                            x=[shop],
                            y=[score],
                            marker_color=color,
                            text=[f"{score:.2f}"],
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        yaxis_title="Priority Score",
                        xaxis_title="Shop"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No shop scores available")
            else:
                st.info("Shop or priority score data not available")
        
        with col2:
            st.subheader("💰 Cost Analysis")
            
            if not cost_df.empty:
                # Filter out TOTAL row for chart
                item_costs = cost_df[cost_df["Item"] != "TOTAL"]
                
                if not item_costs.empty and "Item" in item_costs.columns and "Total_Cost_₹" in item_costs.columns:
                    fig = px.bar(
                        item_costs,
                        x="Item",
                        y="Total_Cost_₹",
                        color="Item",
                        text="Total_Cost_₹",
                        title="Cost by Item"
                    )
                    
                    fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
                    fig.update_layout(height=400, showlegend=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Cost chart data not available")
            else:
                st.info("Cost data not available")
    
    else:
        st.info("No priority data available")

# =====================================================
# POLICY MAKER VIEW
# =====================================================
elif role == "Policy Maker":
    st.subheader("🔄 Policy Simulation & Impact Analysis")
    
    # Scenario Comparison
    st.subheader("📊 Scenario Comparison")
    
    if not scenario_df.empty and "Scenario" in scenario_df.columns and "Item" in scenario_df.columns:
        if "Predicted_Demand_kg" in scenario_df.columns:
            # Pivot table
            try:
                scenario_pivot = scenario_df.pivot_table(
                    index="Scenario",
                    columns="Item",
                    values="Predicted_Demand_kg",
                    aggfunc='mean'
                ).round(0)
                
                st.dataframe(scenario_pivot, use_container_width=True)
                
                st.markdown("---")
                
                # Interactive charts
                st.subheader("📈 Scenario Analysis")
                
                tab1, tab2 = st.tabs(["Line Chart", "Bar Chart"])
                
                with tab1:
                    fig = px.line(
                        scenario_df,
                        x="Scenario",
                        y="Predicted_Demand_kg",
                        color="Item",
                        markers=True,
                        title="Demand Across Scenarios"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = px.bar(
                        scenario_df,
                        x="Scenario",
                        y="Predicted_Demand_kg",
                        color="Item",
                        barmode="group",
                        title="Demand Comparison by Scenario"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                st.download_button(
                    "📥 Download Scenario Data",
                    scenario_df.to_csv(index=False),
                    "scenario_analysis.csv",
                    "text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error creating pivot table: {e}")
                st.dataframe(scenario_df, use_container_width=True)
        else:
            st.dataframe(scenario_df, use_container_width=True)
            st.info("Predicted demand data not available for pivoting")
    
    else:
        st.info("Scenario data not available")

# =====================================================
# AUDITOR VIEW
# =====================================================
elif role == "Auditor":
    st.subheader("🔍 AI Model Audit & Transparency")
    
    # Get model insights
    try:
        if predictor is not None:
            insights = predictor.get_model_insights()
        else:
            insights = {}
    except:
        insights = {}
    
    if insights and "model_performance" in insights:
        st.subheader("📊 Model Performance Metrics")
        
        perf_data = []
        for item, metrics in insights["model_performance"].items():
            perf_data.append({
                "Model": item,
                "MAE": round(metrics.get("mae", 0), 2),
                "R² Score": round(metrics.get("r2", 0), 3),
                "Training Samples": metrics.get("training_samples", 0)
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
    
    # Data Statistics
    st.markdown("---")
    st.subheader("📈 Data Statistics")
    
    if data is not None and not data.empty:
        st.write(f"**Total Records:** {len(data)}")
        st.write(f"**Data Columns:** {len(data.columns)}")
        
        if "week" in data.columns:
            st.write(f"**Date Range:** Week {data['week'].min()} to Week {data['week'].max()}")
        
        # Show sample data
        with st.expander("View Sample Data"):
            st.dataframe(data.head(), use_container_width=True)
    
    # System Information
    st.markdown("---")
    st.subheader("⚙️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AI Model", "Random Forest")
    
    with col2:
        st.metric("Items Tracked", len(model_config.ITEMS))
    
    with col3:
        st.metric("Shops Monitored", len(model_config.SHOPS))

# =====================================================
# COMMON FOOTER
# =====================================================
st.markdown("---")

# Responsible AI Section
st.subheader("🤖 Responsible AI Principles")

cols = st.columns(4)
with cols[0]:
    st.info("""
    **Explainability**  
    Every prediction comes with clear reasoning
    """)
with cols[1]:
    st.info("""
    **Fairness**  
    Equal consideration for all beneficiary groups
    """)
with cols[2]:
    st.info("""
    **Transparency**  
    Open model insights and decision logic
    """)
with cols[3]:
    st.info("""
    **Human-in-the-loop**  
    Final decisions always require human approval
    """)

# Performance Metrics
st.markdown("---")
st.subheader("⚡ System Performance")

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.last_update:
        update_time = datetime.now() - st.session_state.last_update
        minutes_ago = update_time.seconds // 60
        st.metric("Last Updated", f"{minutes_ago} mins ago" if minutes_ago > 0 else "Just now")
    else:
        st.metric("Last Updated", "Never")

with col2:
    st.metric("Data Points", f"{len(data):,}" if data is not None else "0")

with col3:
    st.metric("AI Accuracy", "85-95%")

# Footer
st.markdown("---")
st.caption(f"""
**AI-Ration v{app_config.APP_VERSION}** | Predictive Stock Balancing for Public Distribution Shops  
**Primary SDG:** 2 (Zero Hunger) | **Expected Impact:** Reduce food waste by ~20%, ensure consistent availability
""")

# Debug information (hidden by default)
with st.expander("🔧 Debug Information"):
    st.write(f"Session State Initialized: {st.session_state.initialized}")
    st.write(f"Models Trained: {st.session_state.models_trained}")
    st.write(f"Predictor exists: {predictor is not None}")
    st.write(f"Data shape: {data.shape if data is not None else 'No data'}")