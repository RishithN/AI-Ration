"""
AI-Ration: Core Machine Learning Engine
Fixed version with proper initialization
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from config import model_config, app_config

class AIDemandPredictor:
    """
    Enhanced demand prediction engine with multiple algorithms
    """
    
    def __init__(self, model_type: str = "ensemble"):
        """
        Initialize the predictor
        
        Args:
            model_type: Type of model to use ('linear', 'rf', 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metadata = {}
        self.item_stats = {}  # Initialize here
        self.data = None  # Store data reference
        
    def create_synthetic_data(self, weeks: int = 52) -> pd.DataFrame:
        """
        Create more realistic synthetic data with seasonal patterns
        """
        np.random.seed(42)
        
        data = {
            "week": list(range(1, weeks + 1)),
            "month": [((i-1) % 12) + 1 for i in range(1, weeks + 1)],
            "is_festival_season": [],
            "is_harvest_season": [],
            "temperature": [],
            "rainfall": []
        }
        
        # Generate item sales with trends and noise
        items = {}
        for item in model_config.ITEMS.keys():
            base_sales = {
                "Rice": 520,
                "Wheat": 420,
                "Sugar": 150
            }[item]
            
            # Add seasonal patterns
            sales = []
            for week in range(weeks):
                month = data["month"][week]
                
                # Base with trend
                base = base_sales * (1 + 0.01 * week)  # Small upward trend
                
                # Seasonal effects
                if month in [10, 11]:  # Festival season
                    seasonal_factor = 1.2
                elif month in [6, 7]:  # Rainy season
                    seasonal_factor = 1.1
                elif month in [3, 4]:  # Harvest season
                    seasonal_factor = 0.9
                else:
                    seasonal_factor = 1.0
                
                # Add randomness
                noise = np.random.normal(0, 0.1)
                final_sales = base * seasonal_factor * (1 + noise)
                sales.append(max(100, final_sales))  # Ensure positive
                
            data[f"{item.lower()}_sold"] = [round(s, 2) for s in sales]
        
        # Generate context features
        for week in range(weeks):
            month = data["month"][week]
            
            # Festival season (Oct-Nov)
            data["is_festival_season"].append(1 if month in [10, 11] else 0)
            
            # Harvest season (Mar-Apr)
            data["is_harvest_season"].append(1 if month in [3, 4] else 0)
            
            # Simulated temperature
            temp = 25 + 10 * np.sin(2 * np.pi * week / 52)
            data["temperature"].append(round(temp + np.random.normal(0, 3), 1))
            
            # Simulated rainfall (monsoon: Jun-Sep)
            if month in [6, 7, 8, 9]:
                rainfall = np.random.exponential(5)
            else:
                rainfall = np.random.exponential(0.5)
            data["rainfall"].append(round(rainfall, 1))
        
        # Add wage weeks (last week of each month)
        data["wage_week"] = [(1 if week % 4 == 0 else 0) for week in range(weeks)]
        
        # Add festival weeks (random but clustered)
        data["festival_week"] = []
        festival_prob = 0.1
        for week in range(weeks):
            if data["is_festival_season"][week]:
                festival_prob = 0.3
            else:
                festival_prob = 0.05
            data["festival_week"].append(1 if np.random.random() < festival_prob else 0)
        
        # Add rainy weeks
        data["rainy_week"] = [(1 if r > 2 else 0) for r in data["rainfall"]]
        
        df = pd.DataFrame(data)
        self.data = df  # Store reference
        return df
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from file or generate synthetic data
        """
        data_file = app_config.DATA_DIR / "pds_data.csv"
        
        if data_file.exists():
            df = pd.read_csv(data_file)
        else:
            df = self.create_synthetic_data()
            df.to_csv(data_file, index=False)
            
        self.data = df  # Store reference
        return df
    
    def prepare_features(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare features for model training
        """
        if df is None:
            if self.data is None:
                df = self.load_data()
            else:
                df = self.data
        else:
            self.data = df
        
        features = []
        
        for _, row in df.iterrows():
            for item in model_config.ITEMS.keys():
                feature = {
                    "item": item,
                    "week": row["week"],
                    "month": row["month"],
                    "festival_week": row["festival_week"],
                    "wage_week": row["wage_week"],
                    "rainy_week": row["rainy_week"],
                    "is_festival_season": row["is_festival_season"],
                    "is_harvest_season": row["is_harvest_season"],
                    "temperature": row["temperature"],
                    "rainfall": row["rainfall"],
                    "target": row[f"{item.lower()}_sold"]
                }
                features.append(feature)
        
        features_df = pd.DataFrame(features)
        
        # Calculate item statistics
        self.item_stats = {}  # Reset and recalculate
        for item in model_config.ITEMS.keys():
            item_data = features_df[features_df["item"] == item]
            self.item_stats[item] = {
                "mean": item_data["target"].mean(),
                "std": item_data["target"].std(),
                "min": item_data["target"].min(),
                "max": item_data["target"].max(),
                "median": item_data["target"].median()
            }
        
        return features_df, self.item_stats
    
    def train_models(self, df: pd.DataFrame = None) -> Dict:
        """
        Train multiple models for each item
        """
        if df is not None:
            self.data = df
        
        features_df, item_stats = self.prepare_features(df)
        self.item_stats = item_stats  # Ensure item_stats is set
        
        for item in model_config.ITEMS.keys():
            # Prepare data for this item
            item_data = features_df[features_df["item"] == item]
            X = item_data.drop(["item", "target"], axis=1)
            y = item_data["target"]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model based on type
            if self.model_type == "linear":
                model = LinearRegression()
            elif self.model_type == "rf":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
            else:  # ensemble - use both
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
            
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self.models[item] = model
            self.scalers[item] = scaler
            
            # Calculate feature importance
            if hasattr(model, "feature_importances_"):
                importance = dict(zip(X.columns, model.feature_importances_))
            else:
                importance = dict(zip(X.columns, np.abs(model.coef_)))
            
            self.feature_importance[item] = importance
            
            # Calculate model performance
            predictions = model.predict(X_scaled)
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            self.model_metadata[item] = {
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "training_samples": len(y)
            }
        
        self.save_models()
        return self.models
    
    def predict(self, models: Dict = None, context: Dict = None) -> pd.DataFrame:
        """
        Predict demand for all items given context
        """
        if models is None:
            models = self.models
            
        if context is None:
            context = {}
            
        # Ensure item_stats exists
        if not hasattr(self, 'item_stats') or not self.item_stats:
            if self.data is not None:
                self.prepare_features(self.data)
            else:
                # Create default item_stats
                self.item_stats = {}
                for item in model_config.ITEMS.keys():
                    self.item_stats[item] = {
                        "mean": 500,
                        "std": 100,
                        "min": 100,
                        "max": 1000
                    }
        
        predictions = []
        
        for item, model in models.items():
            # Prepare input features
            input_features = {
                "week": 53,  # Next week
                "month": context.get("month", (datetime.now().month % 12) + 1),
                "festival_week": context.get("festival_week", 0),
                "wage_week": context.get("wage_week", 0),
                "rainy_week": context.get("rainy_week", 0),
                "is_festival_season": context.get("is_festival_season", 0),
                "is_harvest_season": context.get("is_harvest_season", 0),
                "temperature": context.get("temperature", 25.0),
                "rainfall": context.get("rainfall", 0.0)
            }
            
            # Convert to DataFrame for scaling
            input_df = pd.DataFrame([input_features])
            
            # Scale features
            if item in self.scalers:
                input_scaled = self.scalers[item].transform(input_df)
            else:
                input_scaled = input_df.values
            
            # Make prediction
            pred_value = model.predict(input_scaled)[0]
            
            # Add confidence interval based on historical variability
            if item in self.item_stats:
                stats = self.item_stats[item]
                std = stats.get("std", pred_value * 0.1)
                lower_bound = max(0, pred_value - 1.96 * std / np.sqrt(12))
                upper_bound = pred_value + 1.96 * std / np.sqrt(12)
            else:
                lower_bound = pred_value * 0.8
                upper_bound = pred_value * 1.2
            
            predictions.append({
                "Item": item,
                "Predicted_Demand_kg": round(pred_value, 2),
                "Lower_Bound_kg": round(lower_bound, 2),
                "Upper_Bound_kg": round(upper_bound, 2),
                "Confidence_Interval": f"{round(lower_bound, 0)} - {round(upper_bound, 0)}"
            })
        
        return pd.DataFrame(predictions)
    
    def apply_decision_logic(self, pred_df: pd.DataFrame, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Apply business logic to convert predictions into actionable insights
        """
        if df is None:
            df = self.data
            
        # Ensure item_stats exists
        if not hasattr(self, 'item_stats') or not self.item_stats:
            self.prepare_features(df)
        
        results = []
        
        for _, row in pred_df.iterrows():
            item = row["Item"]
            pred = row["Predicted_Demand_kg"]
            
            # Get historical statistics
            if item in self.item_stats:
                stats = self.item_stats[item]
                avg = stats["mean"]
                std = stats["std"]
            else:
                # Fallback: calculate from data
                col_name = f"{item.lower()}_sold"
                if col_name in df.columns:
                    avg = df[col_name].mean()
                    std = df[col_name].std()
                else:
                    avg = pred
                    std = pred * 0.1
            
            # Calculate z-score
            z_score = (pred - avg) / std if std > 0 else 0
            
            # Determine risk level
            if pred > avg * model_config.THRESHOLDS["CRITICAL"]:
                risk = "🔴 Critical Shortage"
                risk_level = "critical"
                color = "#F44336"
                action = "Immediate stock increase (25%+)"
            elif pred > avg * model_config.THRESHOLDS["HIGH_DEMAND"]:
                risk = "🟠 High Demand"
                risk_level = "high"
                color = "#FF9800"
                action = "Increase procurement by 15-20%"
            elif pred < avg * model_config.THRESHOLDS["SEVERE_SHORTAGE"]:
                risk = "⚫ Severe Overstock"
                risk_level = "overstock"
                color = "#607D8B"
                action = "Reduce procurement by 25%"
            elif pred < avg * model_config.THRESHOLDS["LOW_DEMAND"]:
                risk = "🟡 Moderate"
                risk_level = "moderate"
                color = "#FFD700"
                action = "Reduce procurement by 10%"
            else:
                risk = "🟢 Normal"
                risk_level = "normal"
                color = "#4CAF50"
                action = "Maintain current stock levels"
            
            # Calculate recommended order quantity
            base_stock = model_config.ITEMS[item]["safety_stock"]
            if risk_level in ["critical", "high"]:
                recommended_qty = pred * 1.2
            elif risk_level == "overstock":
                recommended_qty = pred * 0.75
            elif risk_level == "moderate":
                recommended_qty = pred * 0.9
            else:
                recommended_qty = pred
            
            # Ensure within capacity limits
            max_capacity = model_config.ITEMS[item]["max_capacity"]
            recommended_qty = min(recommended_qty, max_capacity)
            
            results.append({
                "Item": item,
                "Predicted_Demand_kg": round(pred, 2),
                "Historical_Avg_kg": round(avg, 2),
                "Deviation_%": round(((pred - avg) / avg * 100), 1),
                "Z_Score": round(z_score, 2),
                "Risk": risk,
                "Risk_Color": color,
                "Action_Required": action,
                "Recommended_Order_kg": round(recommended_qty, 2),
                "Current_Stock_kg": round(base_stock * 0.7, 2),  # Simulated
                "Stock_Out_Days": max(0, round((recommended_qty - pred) / pred * 7, 1)),
                "Confidence_Interval": row["Confidence_Interval"]
            })
        
        return pd.DataFrame(results)
    
    def district_prioritization(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prioritize shops based on risk and impact
        """
        rows = []
        
        for shop_name, shop_info in model_config.SHOPS.items():
            shop_weight = shop_info["impact_weight"]
            
            for _, row in final_df.iterrows():
                item = row["Item"]
                risk = row["Risk"]
                
                # Get risk score
                risk_score = model_config.RISK_SCORES.get(risk, 0.5)
                
                # Adjust based on shop characteristics
                if shop_info["location"] == "Urban":
                    location_factor = 1.2
                elif shop_info["location"] == "Rural":
                    location_factor = 0.9
                else:
                    location_factor = 1.0
                
                # Calculate priority score
                priority_score = risk_score * shop_weight * location_factor
                
                # Determine priority level
                if priority_score >= 2.0:
                    priority_level = "🔴 Critical"
                    action = "Immediate Dispatch"
                elif priority_score >= 1.5:
                    priority_level = "🟠 High"
                    action = "Priority Dispatch"
                elif priority_score >= 1.0:
                    priority_level = "🟡 Medium"
                    action = "Schedule Dispatch"
                else:
                    priority_level = "🟢 Low"
                    action = "Regular Schedule"
                
                rows.append({
                    "Shop": shop_name,
                    "Location": shop_info["location"],
                    "Beneficiaries": shop_info["beneficiaries"],
                    "Item": item,
                    "Risk": risk,
                    "Impact_Weight": shop_weight,
                    "Priority_Score": round(priority_score, 2),
                    "Priority_Level": priority_level,
                    "Recommended_Action": action,
                    "Urgency": "Immediate" if priority_score >= 2.0 else 
                              "High" if priority_score >= 1.5 else
                              "Medium" if priority_score >= 1.0 else "Low"
                })
        
        return pd.DataFrame(rows).sort_values("Priority_Score", ascending=False)
    
    def cost_optimization(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize procurement costs based on recommendations
        """
        rows = []
        total_cost = 0
        
        for _, row in final_df.iterrows():
            item = row["Item"]
            recommended_qty = row["Recommended_Order_kg"]
            risk = row["Risk"]
            
            # Get item cost and properties
            item_info = model_config.ITEMS[item]
            cost_per_kg = item_info["base_price"]
            
            # Calculate cost with adjustments
            base_cost = recommended_qty * cost_per_kg
            
            # Apply risk-based adjustments
            if "Critical" in risk or "High Demand" in risk:
                # Rush ordering premium
                cost_multiplier = 1.1
                logistics = "Express Delivery"
            elif "Overstock" in risk:
                # Discount for reduced order
                cost_multiplier = 0.95
                logistics = "Standard Delivery"
            else:
                cost_multiplier = 1.0
                logistics = "Regular Schedule"
            
            adjusted_cost = base_cost * cost_multiplier
            
            # Calculate storage cost (simplified)
            storage_days = 7  # Average storage period
            storage_cost_per_day = 0.01  # 1% per day
            storage_cost = adjusted_cost * storage_days * storage_cost_per_day
            
            total_item_cost = adjusted_cost + storage_cost
            
            rows.append({
                "Item": item,
                "Recommended_Qty_kg": round(recommended_qty, 2),
                "Cost_per_kg_₹": cost_per_kg,
                "Base_Cost_₹": round(base_cost, 2),
                "Risk_Adjustment_%": round((cost_multiplier - 1) * 100, 1),
                "Storage_Cost_₹": round(storage_cost, 2),
                "Logistics": logistics,
                "Total_Cost_₹": round(total_item_cost, 2),
                "Cost_per_Beneficiary_₹": round(total_item_cost / 1000, 2),  # Assuming 1000 beneficiaries
                "Budget_Utilization_%": min(100, round((total_item_cost / 50000) * 100, 1))  # Assuming 50k budget
            })
            
            total_cost += total_item_cost
        
        # Add summary row
        summary_row = {
            "Item": "TOTAL",
            "Recommended_Qty_kg": sum(r["Recommended_Qty_kg"] for r in rows),
            "Cost_per_kg_₹": round(total_cost / sum(r["Recommended_Qty_kg"] for r in rows), 2),
            "Base_Cost_₹": sum(r["Base_Cost_₹"] for r in rows),
            "Risk_Adjustment_%": 0,
            "Storage_Cost_₹": sum(r["Storage_Cost_₹"] for r in rows),
            "Logistics": "Mixed",
            "Total_Cost_₹": round(total_cost, 2),
            "Cost_per_Beneficiary_₹": round(total_cost / 3000, 2),  # Total beneficiaries
            "Budget_Utilization_%": min(100, round((total_cost / 150000) * 100, 1))  # Total budget
        }
        
        result_df = pd.DataFrame(rows)
        result_df.loc[len(result_df)] = summary_row
        
        return result_df
    
    def scenario_simulation(self, models: Dict = None, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run multiple what-if scenarios for policy analysis
        """
        if models is None:
            models = self.models
            
        if df is None:
            df = self.data
            
        scenarios = {
            "Normal": {
                "festival_week": 0,
                "wage_week": 0,
                "rainy_week": 0,
                "is_festival_season": 0,
                "temperature": 25.0,
                "rainfall": 0.0
            },
            "Festival Peak": {
                "festival_week": 1,
                "wage_week": 1,
                "rainy_week": 0,
                "is_festival_season": 1,
                "temperature": 28.0,
                "rainfall": 0.0
            },
            "Monsoon": {
                "festival_week": 0,
                "wage_week": 0,
                "rainy_week": 1,
                "is_festival_season": 0,
                "temperature": 22.0,
                "rainfall": 8.0
            },
            "Heatwave": {
                "festival_week": 0,
                "wage_week": 1,
                "rainy_week": 0,
                "is_festival_season": 0,
                "temperature": 38.0,
                "rainfall": 0.0
            },
            "Payday + Festival": {
                "festival_week": 1,
                "wage_week": 1,
                "rainy_week": 0,
                "is_festival_season": 1,
                "temperature": 26.0,
                "rainfall": 0.0
            },
            "Lockdown": {
                "festival_week": 0,
                "wage_week": 0,
                "rainy_week": 0,
                "is_festival_season": 0,
                "temperature": 25.0,
                "rainfall": 0.0
            }
        }
        
        rows = []
        
        for scenario_name, context in scenarios.items():
            # Predict for this scenario
            pred_df = self.predict(models, context)
            
            # Apply decision logic
            scenario_results = self.apply_decision_logic(pred_df, df)
            
            for _, row in scenario_results.iterrows():
                rows.append({
                    "Scenario": scenario_name,
                    "Item": row["Item"],
                    "Predicted_Demand_kg": row["Predicted_Demand_kg"],
                    "Risk": row["Risk"],
                    "Action_Required": row["Action_Required"],
                    "Recommended_Order_kg": row["Recommended_Order_kg"],
                    "Deviation_%": row["Deviation_%"],
                    "Context_Festival": context["festival_week"],
                    "Context_Wage": context["wage_week"],
                    "Context_Rainy": context["rainy_week"],
                    "Context_Temp": context["temperature"]
                })
        
        return pd.DataFrame(rows)
    
    def get_model_insights(self) -> Dict:
        """
        Get insights about the trained models
        """
        # Ensure attributes exist
        if not hasattr(self, 'model_metadata'):
            self.model_metadata = {}
        if not hasattr(self, 'feature_importance'):
            self.feature_importance = {}
        if not hasattr(self, 'item_stats'):
            self.item_stats = {}
            
        insights = {
            "model_performance": self.model_metadata,
            "feature_importance": self.feature_importance,
            "item_statistics": self.item_stats,
            "training_date": datetime.now().isoformat(),
            "model_type": self.model_type,
            "number_of_models": len(self.models)
        }
        return insights
    
    def save_models(self):
        """
        Save trained models to disk
        """
        # Ensure all attributes exist
        if not hasattr(self, 'item_stats'):
            self.item_stats = {}
        if not hasattr(self, 'feature_importance'):
            self.feature_importance = {}
        if not hasattr(self, 'model_metadata'):
            self.model_metadata = {}
            
        save_data = {
            "models": self.models,
            "scalers": self.scalers,
            "item_stats": self.item_stats,
            "feature_importance": self.feature_importance,
            "model_metadata": self.model_metadata,
            "data_columns": list(self.data.columns) if self.data is not None else []
        }
        
        joblib.dump(save_data, app_config.MODELS_DIR / "ai_ration_models.joblib")
        
        # Also save as JSON for inspection
        json_data = {
            "model_metadata": self.model_metadata,
            "training_date": datetime.now().isoformat(),
            "model_type": self.model_type,
            "items": list(model_config.ITEMS.keys())
        }
        
        with open(app_config.MODELS_DIR / "model_info.json", "w") as f:
            json.dump(json_data, f, indent=2)
    
    def load_models(self) -> bool:
        """
        Load trained models from disk
        """
        try:
            model_path = app_config.MODELS_DIR / "ai_ration_models.joblib"
            if not model_path.exists():
                return False
                
            save_data = joblib.load(model_path)
            self.models = save_data.get("models", {})
            self.scalers = save_data.get("scalers", {})
            self.item_stats = save_data.get("item_stats", {})
            self.feature_importance = save_data.get("feature_importance", {})
            self.model_metadata = save_data.get("model_metadata", {})
            return len(self.models) > 0
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Factory functions for backward compatibility
def load_data():
    """Load data for the dashboard"""
    predictor = AIDemandPredictor()
    return predictor.load_data()

def train_models(df):
    """Train models for the dashboard"""
    predictor = AIDemandPredictor(model_type="rf")
    models = predictor.train_models(df)
    return models

def predict(models, context):
    """Make predictions"""
    predictor = AIDemandPredictor()
    # Try to load existing models first
    if not predictor.load_models():
        # If no saved models, use the passed models
        predictor.models = models
        # We need to create some dummy data to calculate item_stats
        dummy_df = load_data()
        predictor.prepare_features(dummy_df)
    return predictor.predict(models, context)

def apply_decision_logic(pred_df, df):
    """Apply business logic"""
    predictor = AIDemandPredictor()
    predictor.data = df
    predictor.prepare_features(df)
    return predictor.apply_decision_logic(pred_df, df)

def district_prioritization(final_df):
    """Prioritize districts"""
    predictor = AIDemandPredictor()
    return predictor.district_prioritization(final_df)

def cost_optimization(final_df):
    """Optimize costs"""
    predictor = AIDemandPredictor()
    return predictor.cost_optimization(final_df)

def scenario_simulation(models, df):
    """Run scenario simulations"""
    predictor = AIDemandPredictor()
    # Load or set models
    if not predictor.load_models():
        predictor.models = models
    predictor.data = df
    predictor.prepare_features(df)
    return predictor.scenario_simulation(models, df)

def get_model_insights():
    """Get model insights"""
    predictor = AIDemandPredictor()
    if predictor.load_models():
        return predictor.get_model_insights()
    return {}