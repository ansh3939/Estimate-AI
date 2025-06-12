#!/usr/bin/env python3
"""
Setup script for Real Estate Intelligence Platform
Compatible with Python 3.13.5+
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is 3.13.5 or higher"""
    version = sys.version_info
    if version.major != 3 or version.minor < 13 or (version.minor == 13 and version.micro < 5):
        print(f"Error: Python 3.13.5+ required. Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"Python version {version.major}.{version.minor}.{version.micro} is compatible")

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        "streamlit>=1.45.1",
        "pandas>=2.3.0", 
        "numpy>=2.3.0",
        "plotly>=6.1.2",
        "scikit-learn>=1.7.0",
        "sqlalchemy>=2.0.41",
        "psycopg2-binary>=2.9.10",
        "openai>=1.86.0",
        "xgboost>=3.0.2",
        "joblib>=1.5.1",
        "requests>=2.32.4",
        "beautifulsoup4>=4.13.4",
        "trafilatura>=2.0.0"
    ]
    
    print("Installing dependencies...")
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    print("All dependencies installed successfully!")

def setup_environment():
    """Setup environment variables template"""
    env_template = """# Real Estate Intelligence Platform Environment Variables
# Copy this to .env and fill in your values

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/real_estate_db
PGHOST=localhost
PGPORT=5432
PGUSER=your_username
PGPASSWORD=your_password
PGDATABASE=real_estate_db

# OpenAI API Key (required for AI chatbot)
OPENAI_API_KEY=your_openai_api_key_here
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    
    print("Environment template created: .env.template")
    print("Copy this to .env and configure your settings")

if __name__ == "__main__":
    print("Real Estate Intelligence Platform Setup")
    print("=" * 50)
    
    check_python_version()
    install_dependencies()
    setup_environment()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and configure your settings")
    print("2. Set up PostgreSQL database")
    print("3. Run: streamlit run src/main.py --server.port 5000")