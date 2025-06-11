import pandas as pd
from database import db_manager
import os
from pathlib import Path

def initialize_database():
    """Initialize database with tables and sample data"""
    print("Creating database tables...")
    db_manager.create_tables()
    
    print("Loading CSV data...")
    data_dir = Path("data")
    cities = ["mumbai", "delhi", "gurugram", "noida", "bangalore"]
    
    all_data = []
    
    for city in cities:
        csv_file = data_dir / f"{city}_properties.csv"
        if csv_file.exists():
            print(f"Loading {city} data...")
            df = pd.read_csv(csv_file)
            df['City'] = city.title()
            
            # Ensure all required columns exist
            if 'Price_per_SqFt' not in df.columns:
                df['Price_per_SqFt'] = df['Price_INR'] / df['Area_SqFt']
            
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Importing {len(combined_df)} properties to database...")
        db_manager.import_csv_data(combined_df)
        print("Database initialization completed successfully!")
    else:
        print("No CSV files found to import")

if __name__ == "__main__":
    initialize_database()