"""
Authentication utilities for AI-Ration
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import streamlit as st

class Authentication:
    """Simple authentication system for demo purposes"""
    
    # Demo users (In production, use a proper database)
    DEMO_USERS = {
        "shopkeeper": {
            "password": "shop123",
            "role": "Shopkeeper",
            "shop_id": "SHOP_001",
            "permissions": ["view_forecast", "download_reports"]
        },
        "admin": {
            "password": "admin123",
            "role": "Admin",
            "permissions": ["all"]
        },
        "policymaker": {
            "password": "policy123",
            "role": "Policy Maker",
            "permissions": ["view_scenarios", "download_insights"]
        },
        "auditor": {
            "password": "audit123",
            "role": "Auditor",
            "permissions": ["view_metrics", "audit_models"]
        }
    }
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Dict]:
        """Authenticate a user"""
        if username in Authentication.DEMO_USERS:
            user = Authentication.DEMO_USERS[username]
            # In demo, compare plain text. In production, compare hashes
            if password == user["password"]:
                return {
                    "username": username,
                    "role": user["role"],
                    "permissions": user["permissions"],
                    "authenticated_at": datetime.now().isoformat()
                }
        return None
    
    @staticmethod
    def check_permission(required_permission: str) -> bool:
        """Check if current user has required permission"""
        if "user_info" in st.session_state:
            user_info = st.session_state.user_info
            if "all" in user_info.get("permissions", []):
                return True
            return required_permission in user_info.get("permissions", [])
        return False