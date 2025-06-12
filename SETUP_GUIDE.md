# Real Estate Intelligence Platform - Setup Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.11 or higher
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB free space
- **Internet**: Required for AI features and database connection

### Recommended Requirements
- **Python**: 3.11+ for optimal performance
- **RAM**: 8 GB or more
- **Storage**: 5 GB free space
- **Internet**: Stable broadband connection

---

## Installation Guide

### For macOS Users

#### Step 1: Install Python 3.11+
```bash
# Option 1: Using Homebrew (Recommended)
brew install python@3.11

# Option 2: Download from python.org
# Visit https://www.python.org/downloads/mac-osx/
# Download Python 3.11+ installer and run it
```

#### Step 2: Verify Python Installation
```bash
python3 --version
# Should show Python 3.11.x or higher
```

#### Step 3: Clone/Download Project
```bash
# If you have the project files, navigate to the directory
cd /path/to/real-estate-platform

# Or download the project files to your desired location
```

#### Step 4: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

#### Step 5: Install Dependencies
```bash
# Install required packages
pip install streamlit==1.45.1
pip install pandas==2.3.0
pip install numpy==2.3.0
pip install plotly==6.1.2
pip install scikit-learn==1.7.0
pip install sqlalchemy==2.0.41
pip install psycopg2-binary==2.9.10
pip install openai==1.86.0
pip install xgboost==3.0.2
pip install joblib==1.5.1
pip install requests==2.32.4
pip install beautifulsoup4==4.13.4
pip install trafilatura==2.0.0

# Or install all at once
pip install streamlit pandas numpy plotly scikit-learn sqlalchemy psycopg2-binary openai xgboost joblib requests beautifulsoup4 trafilatura
```

#### Step 6: Run the Application
```bash
# Make sure you're in the project directory with main.py
streamlit run main.py

# The webapp will open automatically in your default browser
# If not, go to http://localhost:8501
```

---

### For Windows Users

#### Step 1: Install Python 3.11+
1. Visit https://www.python.org/downloads/windows/
2. Download Python 3.11+ installer
3. **Important**: Check "Add Python to PATH" during installation
4. Run the installer as Administrator

#### Step 2: Verify Python Installation
```cmd
# Open Command Prompt or PowerShell
python --version
# Should show Python 3.11.x or higher

# If python command doesn't work, try:
python3 --version
py --version
```

#### Step 3: Navigate to Project Directory
```cmd
# Open Command Prompt and navigate to your project folder
cd C:\path\to\real-estate-platform

# Or use PowerShell
# Or open the folder in File Explorer, type 'cmd' in address bar
```

#### Step 4: Create Virtual Environment
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your command prompt
```

#### Step 5: Install Dependencies
```cmd
# Install required packages
pip install streamlit==1.45.1
pip install pandas==2.3.0
pip install numpy==2.3.0
pip install plotly==6.1.2
pip install scikit-learn==1.7.0
pip install sqlalchemy==2.0.41
pip install psycopg2-binary==2.9.10
pip install openai==1.86.0
pip install xgboost==3.0.2
pip install joblib==1.5.1
pip install requests==2.32.4
pip install beautifulsoup4==4.13.4
pip install trafilatura==2.0.0

# Or install all dependencies at once
pip install streamlit pandas numpy plotly scikit-learn sqlalchemy psycopg2-binary openai xgboost joblib requests beautifulsoup4 trafilatura
```

#### Step 6: Run the Application
```cmd
# Make sure you're in the project directory with main.py
streamlit run main.py

# The webapp will open automatically in your default browser
# If not, go to http://localhost:8501
```

---

## Environment Setup

### Database Configuration
The webapp connects to a PostgreSQL database. You have two options:

#### Option 1: Use Existing Cloud Database (Recommended)
The webapp is pre-configured to use the existing Neon PostgreSQL database with 1,377 property records.

#### Option 2: Set Up Local Database
If you want to run with a local database:
```bash
# Install PostgreSQL locally
# macOS: brew install postgresql
# Windows: Download from https://www.postgresql.org/download/windows/

# Create database and import data
createdb real_estate_db
# Import the dataset from complete_property_dataset.csv
```

### API Configuration
For AI chatbot functionality, you'll need an OpenAI API key:

1. Visit https://platform.openai.com/
2. Create account and get API key
3. Set environment variable:

**macOS/Linux:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Windows:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

---

## Running the Application

### Starting the Webapp
```bash
# Activate virtual environment first
# macOS: source venv/bin/activate
# Windows: venv\Scripts\activate

# Run the application
streamlit run main.py

# Access at: http://localhost:8501
```

### Stopping the Application
- Press `Ctrl+C` in the terminal
- Or close the terminal window

### Accessing from Other Devices
To access from other devices on the same network:
```bash
# Run with network access
streamlit run main.py --server.address 0.0.0.0

# Then access from other devices using:
# http://YOUR_COMPUTER_IP:8501
```

---

## Troubleshooting

### Common Issues

#### Python Not Found
**Windows:**
- Reinstall Python and check "Add to PATH"
- Use `py` instead of `python`

**macOS:**
- Use `python3` instead of `python`
- Install via Homebrew if issues persist

#### Package Installation Errors
```bash
# Upgrade pip first
pip install --upgrade pip

# Install packages one by one if bulk install fails
pip install streamlit
pip install pandas
# etc.
```

#### Port Already in Use
```bash
# Use different port
streamlit run main.py --server.port 8502

# Or kill existing process
# macOS/Linux: lsof -ti:8501 | xargs kill -9
# Windows: netstat -ano | findstr :8501
```

#### Database Connection Issues
- Check internet connection
- Verify DATABASE_URL is set correctly
- Contact administrator if persistent issues

#### OpenAI API Issues
- Verify API key is correct
- Check account has available credits
- The app will use fallback responses if API unavailable

### Performance Optimization
```bash
# For better performance, install optional dependencies
pip install numba  # Faster numerical computations
pip install lxml   # Faster XML parsing
```

---

## Features Overview

### Complete Features Available:
1. **Property Price Prediction** - AI-powered price estimation
2. **EMI Calculator** - Loan payment calculations
3. **Portfolio Tracker** - Investment portfolio management
4. **AI Chatbot Assistant** - Real estate advice and guidance
5. **Investment Analyzer** - ROI and investment scoring
6. **Market Trends** - Historical analysis and projections
7. **Prediction History** - User prediction tracking

### Data Coverage:
- **1,377 verified properties** across 25 Indian cities
- **Metro cities**: Mumbai, Bangalore, Delhi, Gurugram, Noida
- **Tier-2 cities**: Pune, Chennai, Ahmedabad, Hyderabad, etc.
- **Price ranges**: ₹20 Lakhs to ₹50+ Crores
- **Property types**: Apartments, Villas, Independent Houses

---

## Next Steps

1. **Install Python 3.11+** on your system
2. **Download/clone** the project files
3. **Set up virtual environment** and install dependencies
4. **Run the application** with `streamlit run main.py`
5. **Access the webapp** at http://localhost:8501
6. **Explore all features** and test predictions

### Optional Enhancements:
- Set up OpenAI API key for enhanced chatbot functionality
- Configure local database for offline usage
- Customize settings for your specific needs

The webapp is **complete and fully functional** with all core features implemented and tested.