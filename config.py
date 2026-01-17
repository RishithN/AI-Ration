"""
Configuration settings for AI-Ration
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

@dataclass
class AppConfig:
    """Application configuration"""
    APP_NAME = "AI-Ration: Predictive Stock Balancing"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Create directories
    for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        dir_path.mkdir(exist_ok=True)

@dataclass
class ModelConfig:
    """Model configuration"""
    # Items and their properties
    ITEMS = {
        "Rice": {
            "base_price": 28.0,
            "unit": "kg",
            "safety_stock": 50.0,
            "max_capacity": 1000.0
        },
        "Wheat": {
            "base_price": 24.0,
            "unit": "kg",
            "safety_stock": 40.0,
            "max_capacity": 800.0
        },
        "Sugar": {
            "base_price": 32.0,
            "unit": "kg",
            "safety_stock": 30.0,
            "max_capacity": 500.0
        }
    }
    
    # Shops and their properties
    SHOPS = {
        "Shop_A": {
            "location": "Urban",
            "beneficiaries": 500,
            "impact_weight": 1.3
        },
        "Shop_B": {
            "location": "Rural",
            "beneficiaries": 300,
            "impact_weight": 1.0
        },
        "Shop_C": {
            "location": "Semi-Urban",
            "beneficiaries": 400,
            "impact_weight": 1.1
        }
    }
    
    # Risk thresholds
    THRESHOLDS = {
        "HIGH_DEMAND": 1.15,  # 15% above average
        "LOW_DEMAND": 0.85,   # 15% below average
        "CRITICAL": 1.25,     # 25% above average
        "SEVERE_SHORTAGE": 0.75  # 25% below average
    }
    
    # Priority scoring
    RISK_SCORES = {
        "🟢 Normal": 0.5,
        "🟡 Moderate": 1.0,
        "🟠 High Demand": 1.5,
        "🔴 Critical Shortage": 2.0,
        "⚫ Severe Overstock": 0.3
    }

@dataclass
class UIConfig:
    """UI configuration"""
    PRIMARY_COLOR = "#2E86AB"
    SECONDARY_COLOR = "#A23B72"
    SUCCESS_COLOR = "#4CAF50"
    WARNING_COLOR = "#FF9800"
    DANGER_COLOR = "#F44336"
    INFO_COLOR = "#2196F3"
    
    # Chart colors
    CHART_COLORS = {
        "Rice": "#FF6B6B",
        "Wheat": "#4ECDC4",
        "Sugar": "#FFD166",
        "Normal": "#4CAF50",
        "Warning": "#FF9800",
        "Critical": "#F44336"
    }
    
    # Layout
    PAGE_LAYOUT = "wide"
    SIDEBAR_COLLAPSED = False

# Global configuration instances
app_config = AppConfig()
model_config = ModelConfig()
ui_config = UIConfig()