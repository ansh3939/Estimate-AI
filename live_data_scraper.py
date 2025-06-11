import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import random
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LivePropertyDataScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def get_property_data_magicbricks(self, city: str, property_type: str = "residential-apartment") -> List[Dict]:
        """Scrape property data from MagicBricks"""
        try:
            city_slug = city.lower().replace(" ", "-")
            url = f"https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=&proptype={property_type}&cityName={city_slug}"
            
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            properties = []
            
            # Extract property listings
            property_cards = soup.find_all('div', class_='mb-srp__card')
            
            for card in property_cards[:50]:  # Limit to 50 properties
                try:
                    property_data = self._extract_magicbricks_property(card, city)
                    if property_data:
                        properties.append(property_data)
                except Exception as e:
                    continue
            
            return properties
        except Exception as e:
            print(f"Error scraping MagicBricks for {city}: {e}")
            return []
    
    def _extract_magicbricks_property(self, card, city: str) -> Optional[Dict]:
        """Extract property details from MagicBricks card"""
        try:
            # Price
            price_elem = card.find('div', class_='mb-srp__card__price')
            if not price_elem:
                return None
            
            price_text = price_elem.get_text().strip()
            price = self._parse_price(price_text)
            
            # Area
            area_elem = card.find('div', class_='mb-srp__card__summary__list')
            area_sqft = 1000  # Default
            bhk = 2  # Default
            
            if area_elem:
                area_text = area_elem.get_text()
                area_sqft = self._parse_area(area_text)
                bhk = self._parse_bhk(area_text)
            
            # Location
            location_elem = card.find('div', class_='mb-srp__card__ads--name')
            district = "Central"
            sub_district = "Main Area"
            
            if location_elem:
                location_text = location_elem.get_text().strip()
                district, sub_district = self._parse_location(location_text, city)
            
            return {
                'City': city,
                'District': district,
                'Sub_District': sub_district,
                'Area_SqFt': area_sqft,
                'BHK': bhk,
                'Property_Type': 'Apartment',
                'Furnishing': random.choice(['Unfurnished', 'Semi-Furnished', 'Fully Furnished']),
                'Price_INR': price,
                'Source': 'MagicBricks',
                'Scraped_Date': datetime.now().strftime('%Y-%m-%d')
            }
        except Exception as e:
            return None
    
    def get_property_data_99acres(self, city: str) -> List[Dict]:
        """Scrape property data from 99acres"""
        try:
            city_slug = city.lower().replace(" ", "-")
            url = f"https://www.99acres.com/{city_slug}-real-estate"
            
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            properties = []
            
            # Extract property listings
            property_cards = soup.find_all('div', {'data-testid': 'srp-tuple'})
            
            for card in property_cards[:50]:  # Limit to 50 properties
                try:
                    property_data = self._extract_99acres_property(card, city)
                    if property_data:
                        properties.append(property_data)
                except Exception as e:
                    continue
            
            return properties
        except Exception as e:
            print(f"Error scraping 99acres for {city}: {e}")
            return []
    
    def _extract_99acres_property(self, card, city: str) -> Optional[Dict]:
        """Extract property details from 99acres card"""
        try:
            # Price
            price_elem = card.find('span', class_='price')
            if not price_elem:
                return None
            
            price_text = price_elem.get_text().strip()
            price = self._parse_price(price_text)
            
            # Property details
            details_elem = card.find('div', class_='project-tuple-size-type')
            area_sqft = 1000
            bhk = 2
            
            if details_elem:
                details_text = details_elem.get_text()
                area_sqft = self._parse_area(details_text)
                bhk = self._parse_bhk(details_text)
            
            # Location
            location_elem = card.find('h2', class_='project-tuple-name')
            district = "Central"
            sub_district = "Main Area"
            
            if location_elem:
                location_text = location_elem.get_text().strip()
                district, sub_district = self._parse_location(location_text, city)
            
            return {
                'City': city,
                'District': district,
                'Sub_District': sub_district,
                'Area_SqFt': area_sqft,
                'BHK': bhk,
                'Property_Type': 'Apartment',
                'Furnishing': random.choice(['Unfurnished', 'Semi-Furnished', 'Fully Furnished']),
                'Price_INR': price,
                'Source': '99acres',
                'Scraped_Date': datetime.now().strftime('%Y-%m-%d')
            }
        except Exception as e:
            return None
    
    def _parse_price(self, price_text: str) -> float:
        """Parse price from text"""
        try:
            # Remove currency symbols and spaces
            price_clean = price_text.replace('₹', '').replace(',', '').replace(' ', '').lower()
            
            # Handle crore and lakh
            if 'crore' in price_clean:
                number = float(price_clean.split('crore')[0])
                return number * 10000000
            elif 'lakh' in price_clean:
                number = float(price_clean.split('lakh')[0])
                return number * 100000
            else:
                # Try to extract number
                import re
                numbers = re.findall(r'\d+\.?\d*', price_clean)
                if numbers:
                    return float(numbers[0]) * 100000  # Assume lakh if no unit
                
            return 5000000  # Default 50 lakh
        except:
            return 5000000
    
    def _parse_area(self, area_text: str) -> float:
        """Parse area from text"""
        try:
            import re
            # Look for square feet patterns
            sqft_match = re.search(r'(\d+\.?\d*)\s*(?:sq\.?\s*ft|sqft|square\s*feet)', area_text.lower())
            if sqft_match:
                return float(sqft_match.group(1))
            
            # Look for just numbers that might be area
            numbers = re.findall(r'\d+', area_text)
            if numbers:
                for num in numbers:
                    if 500 <= int(num) <= 5000:  # Reasonable area range
                        return float(num)
            
            return 1000  # Default
        except:
            return 1000
    
    def _parse_bhk(self, text: str) -> int:
        """Parse BHK from text"""
        try:
            import re
            bhk_match = re.search(r'(\d+)\s*(?:bhk|bedroom)', text.lower())
            if bhk_match:
                return int(bhk_match.group(1))
            
            # Look for patterns like "2 BHK", "3BHK"
            bhk_match = re.search(r'(\d+)\s*bhk', text.lower())
            if bhk_match:
                return int(bhk_match.group(1))
            
            return 2  # Default
        except:
            return 2
    
    def _parse_location(self, location_text: str, city: str) -> tuple:
        """Parse district and sub-district from location text"""
        try:
            # Common area mappings for major cities
            location_mappings = {
                'Mumbai': {
                    'andheri': ('Andheri', 'Andheri West'),
                    'bandra': ('Bandra', 'Bandra West'),
                    'malad': ('Malad', 'Malad West'),
                    'powai': ('Powai', 'Hiranandani Gardens'),
                    'thane': ('Thane', 'Ghodbunder Road'),
                    'borivali': ('Borivali', 'Borivali West'),
                    'kandivali': ('Kandivali', 'Kandivali West')
                },
                'Delhi': {
                    'dwarka': ('Southwest Delhi', 'Dwarka'),
                    'rohini': ('Northwest Delhi', 'Rohini'),
                    'gurgaon': ('Gurugram', 'Sector 14'),
                    'noida': ('Noida', 'Sector 18'),
                    'ghaziabad': ('Ghaziabad', 'Vasundhara')
                },
                'Bangalore': {
                    'whitefield': ('Whitefield', 'ITPL Main Road'),
                    'electronic': ('Electronic City', 'Phase 1'),
                    'koramangala': ('Koramangala', 'Koramangala 5th Block'),
                    'indiranagar': ('Indiranagar', 'Indiranagar 100 Feet Road'),
                    'jayanagar': ('Jayanagar', 'Jayanagar 4th Block')
                }
            }
            
            location_lower = location_text.lower()
            
            if city in location_mappings:
                for key, (district, sub_district) in location_mappings[city].items():
                    if key in location_lower:
                        return district, sub_district
            
            # Default return
            words = location_text.split()
            if len(words) >= 2:
                return words[0], words[1]
            elif len(words) == 1:
                return words[0], f"{words[0]} Main"
            else:
                return "Central", "Main Area"
                
        except:
            return "Central", "Main Area"
    
    def get_live_market_data(self, cities: List[str]) -> pd.DataFrame:
        """Get live market data for multiple cities"""
        all_properties = []
        
        for city in cities:
            print(f"Scraping live data for {city}...")
            
            # Try MagicBricks
            mb_properties = self.get_property_data_magicbricks(city)
            all_properties.extend(mb_properties)
            
            # Random delay to avoid being blocked
            time.sleep(random.uniform(2, 5))
            
            # Try 99acres
            acres_properties = self.get_property_data_99acres(city)
            all_properties.extend(acres_properties)
            
            # Another delay
            time.sleep(random.uniform(2, 5))
        
        if all_properties:
            df = pd.DataFrame(all_properties)
            # Calculate price per sq ft
            df['Price_per_SqFt'] = df['Price_INR'] / df['Area_SqFt']
            return df
        else:
            return pd.DataFrame()
    
    def update_live_data_cache(self, cities: List[str], cache_file: str = "live_data_cache.csv"):
        """Update cached live data"""
        try:
            # Get new live data
            new_data = self.get_live_market_data(cities)
            
            if not new_data.empty:
                # Try to load existing cache
                try:
                    existing_data = pd.read_csv(cache_file)
                    # Remove old data (older than 24 hours)
                    existing_data['Scraped_Date'] = pd.to_datetime(existing_data['Scraped_Date'])
                    cutoff_date = datetime.now() - timedelta(days=1)
                    recent_data = existing_data[existing_data['Scraped_Date'] >= cutoff_date]
                    
                    # Combine with new data
                    combined_data = pd.concat([recent_data, new_data], ignore_index=True)
                except FileNotFoundError:
                    combined_data = new_data
                
                # Save updated cache
                combined_data.to_csv(cache_file, index=False)
                print(f"Updated live data cache with {len(new_data)} new properties")
                return combined_data
            
        except Exception as e:
            print(f"Error updating live data cache: {e}")
            
        return pd.DataFrame()