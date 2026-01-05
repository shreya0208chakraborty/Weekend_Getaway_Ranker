#!/usr/bin/env python3
"""
India Travel Recommendation Engine

This script recommends top weekend travel destinations in India based on the user's source city.
It considers distance, rating, and popularity to provide personalized recommendations.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import List, Dict, Tuple, Optional
import os
import sys
from tqdm import tqdm

# Constants
MAX_DISTANCE_KM = 800  # Maximum distance for weekend getaways (increased from 500km)
WEIGHTS = {
    'rating': 0.4,     # 40% weight
    'popularity': 0.3, # 30% weight
    'distance': 0.3    # 30% weight
}

# Major cities with their coordinates
MAJOR_CITIES = {
    'Mumbai': (19.0760, 72.8777),
    'Delhi': (28.6139, 77.2090),
    'Bangalore': (12.9716, 77.5946),
    'Kolkata': (22.5726, 88.3639),
    'Chennai': (13.0827, 80.2707),
    'Hyderabad': (17.3850, 78.4867),
    'Pune': (18.5204, 73.8567),
    'Jaipur': (26.9124, 75.7873),
    'Ahmedabad': (23.0225, 72.5714),
    'Goa': (15.2993, 74.1240),
    'Shimla': (31.1048, 77.1734),
    'Manali': (32.2396, 77.1887),
    'Mysore': (12.2958, 76.6394),
    'Ooty': (11.4102, 76.6950),
    'Kochi': (9.9312, 76.2673),
    'Puri': (19.8135, 85.8312),
    'Vrindavan': (27.5809, 77.7007),
    'Varanasi': (25.3176, 82.9739),
    'Agra': (27.1767, 78.0081),
    'Udaipur': (24.5854, 73.7125)
}

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the travel destinations dataset.
    
    Args:
        filepath: Path to the CSV file containing travel destinations data.
        
    Returns:
        pd.DataFrame: Processed DataFrame containing travel destinations.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Basic data validation
        required_columns = ['City', 'State', 'Google review rating', 'Number of google review in lakhs', 'Type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Debug: Print dataset info
        print("\n=== Dataset Info ===")
        print(f"Total rows: {len(df)}")
        print(f"Unique cities: {df['City'].nunique()}")
        print("\nFirst few city names:", df['City'].head().tolist())
        print("Sample cities with coordinates:")
        print(df[['City', 'State']].drop_duplicates().head(10))
        
        # Clean and prepare the data
        df = df.copy()
        
        # Rename and clean columns
        df = df.rename(columns={
            'Google review rating': 'rating',
            'Number of google review in lakhs': 'review_count_lakhs',
            'Type': 'category'
        })
        
        # Convert review count to numeric and calculate popularity (reviews * rating)
        df['review_count_lakhs'] = pd.to_numeric(df['review_count_lakhs'], errors='coerce')
        df['popularity'] = df['review_count_lakhs'] * df['rating']
        
        # Clean and standardize text data
        df['City'] = df['City'].str.strip().str.title()
        df['State'] = df['State'].str.strip().str.title()
        
        # Convert rating to numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Get unique cities with their average ratings, popularity, and top attraction
        city_data = df.groupby(['City', 'State']).agg({
            'rating': 'mean',
            'popularity': 'sum',
            'category': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown',
            'Name': 'first'  # Get the first attraction name for each city
        }).reset_index()
        
        # Add latitude and longitude using the major cities mapping
        city_data['latitude'] = city_data['City'].map(lambda x: MAJOR_CITIES.get(x, (None, None))[0] if x in MAJOR_CITIES else None)
        city_data['longitude'] = city_data['City'].map(lambda x: MAJOR_CITIES.get(x, (None, None))[1] if x in MAJOR_CITIES else None)
        
        # Drop rows without coordinates (cities not in our MAJOR_CITIES dictionary)
        city_data = city_data.dropna(subset=['latitude', 'longitude'])
        
        return city_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the distance between two coordinates using geodesic distance.
    
    Args:
        coord1: Tuple of (latitude, longitude) for the first point
        coord2: Tuple of (latitude, longitude) for the second point
        
    Returns:
        float: Distance in kilometers
    """
    return geodesic(coord1, coord2).kilometers

def get_city_coordinates(df: pd.DataFrame, city_name: str) -> Optional[Tuple[float, float]]:
    """
    Get the coordinates (latitude, longitude) of a city from the dataset.
    
    Args:
        df: DataFrame containing city data
        city_name: Name of the city to find (case-insensitive)
        
    Returns:
        Optional[Tuple[float, float]]: (latitude, longitude) if found, None otherwise
    """
    # Try to find an exact match first
    city_data = df[df['City'].str.lower() == city_name.lower()]
    
    # If no exact match, try partial match (e.g., 'New Delhi' for 'Delhi')
    if city_data.empty:
        city_data = df[df['City'].str.lower().str.contains(city_name.lower())]
    
    if city_data.empty:
        return None
        
    # Return the first match's coordinates
    return (city_data.iloc[0]['latitude'], city_data.iloc[0]['longitude'])

def normalize_series(series: pd.Series, reverse: bool = False) -> pd.Series:
    """
    Normalize a pandas Series to the range [0, 1].
    
    Args:
        series: Input Series to normalize
        reverse: If True, reverses the normalization (1 - normalized_value)
        
    Returns:
        pd.Series: Normalized Series
    """
    min_val = series.min()
    max_val = series.max()
    
    # Handle case where all values are the same
    if min_val == max_val:
        return pd.Series(1.0, index=series.index)
    
    normalized = (series - min_val) / (max_val - min_val)
    return 1 - normalized if reverse else normalized

def recommend_destinations(
    source_city: str, 
    df: pd.DataFrame, 
    max_distance: float = MAX_DISTANCE_KM,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Recommend top travel destinations from a source city based on multiple factors.
    
    Args:
        source_city: Name of the source city
        df: DataFrame containing travel destinations data
        max_distance: Maximum distance in kilometers for recommendations
        top_n: Number of top recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame containing top recommended destinations
    """
    # Get source city coordinates
    print(f"\n=== Looking for source city: {source_city} ===")
    source_coords = get_city_coordinates(df, source_city)
    if source_coords is None:
        print(f"Error: Source city '{source_city}' not found in the dataset.")
        print("Available cities:", df['City'].unique().tolist())
        return pd.DataFrame()
    print(f"Found coordinates for {source_city}: {source_coords}")
    
    # Create a copy to avoid modifying the original DataFrame
    recommendations = df.copy()
    
    # Calculate distances from source city
    print(f"\nCalculating distances from {source_city}...")
    tqdm.pandas(desc="Processing destinations")
    recommendations['distance_km'] = recommendations.progress_apply(
        lambda row: calculate_distance(
            source_coords, 
            (row['latitude'], row['longitude'])
        ), 
        axis=1
    )
    
    # Filter out the source city and destinations beyond max distance
    before_filter = len(recommendations)
    recommendations = recommendations[
        (recommendations['City'].str.lower() != source_city.lower()) &
        (recommendations['distance_km'] <= max_distance)
    ]
    after_filter = len(recommendations)
    
    print(f"\n=== Distance Filtering ===")
    print(f"Destinations before filtering: {before_filter}")
    print(f"Destinations after filtering: {after_filter} (within {max_distance}km)")
    
    if recommendations.empty:
        print(f"\nNo destinations found within {max_distance} km of {source_city}.")
        print("Closest destinations:")
        closest = df.nsmallest(5, 'distance_km')
        print(closest[['City', 'State', 'distance_km']].to_string(index=False))
        return pd.DataFrame()
    
    # Normalize factors
    recommendations['rating_norm'] = normalize_series(recommendations['rating'])
    recommendations['popularity_norm'] = normalize_series(recommendations['popularity'])
    recommendations['distance_norm'] = normalize_series(recommendations['distance_km'], reverse=True)
    
    # Calculate weighted score
    recommendations['score'] = (
        recommendations['rating_norm'] * WEIGHTS['rating'] +
        recommendations['popularity_norm'] * WEIGHTS['popularity'] +
        recommendations['distance_norm'] * WEIGHTS['distance']
    )
    
    # Sort by score in descending order
    recommendations = recommendations.sort_values('score', ascending=False)
    
    # Select and format the output columns
    result_columns = ['Name', 'City', 'State', 'distance_km', 'rating', 'popularity', 'score', 'category']
    
    # Only include columns that exist in the DataFrame
    result_columns = [col for col in result_columns if col in recommendations.columns]
    
    return recommendations[result_columns].head(top_n).reset_index(drop=True)

def print_recommendations(recommendations: pd.DataFrame, source_city: str) -> None:
    """
    Print the recommendations in a formatted table.
    
    Args:
        recommendations: DataFrame containing recommendations
        source_city: Name of the source city
    """
    if recommendations.empty:
        print("\nNo recommendations found. Please try another city.")
        return
        
    # Debug: Print available columns
    print("\nAvailable columns in recommendations:", recommendations.columns.tolist())
    
    # Format the output
    print(f"\n{'='*100}")
    print(f"TOP {len(recommendations)} WEEKEND GETAWAYS FROM {source_city.upper()}")
    print("="*100)
    
    # Create a copy to avoid modifying the original DataFrame
    display_df = recommendations.copy()
    
    # Format the columns for better display
    display_df['distance_km'] = display_df['distance_km'].round(1).astype(str) + ' km'
    display_df['rating'] = display_df['rating'].round(1).astype(str) + ' ★'
    display_df['popularity'] = (display_df['popularity'] / 1000).round(1).astype(str) + 'K'  # Convert to thousands
    display_df['score'] = (display_df['score'] * 100).round(1).astype(str) + '%'
    
    # Select and reorder columns for display
    display_columns = ['City', 'State', 'Name', 'distance_km', 'category', 'rating', 'popularity', 'score']
    display_df = display_df[display_columns]
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'City': 'City',
        'State': 'State',
        'Name': 'Attraction',
        'distance_km': 'Distance (km)',
        'category': 'Category',
        'rating': 'Rating',
        'popularity': 'Popularity',
        'score': 'Match Score'
    })
    
    # Print the formatted table with adjusted column widths
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.width', 120)
    print(display_df.to_string(index=False, justify='left'))
    print("\n" + "="*100 + "\n")

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, '..', 'data', 'Top_Indian_Places_to_Visit.csv')
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure the file 'india_travel_places.csv' exists in the data directory.")
        sys.exit(1)
    
    # Load the data
    print("Loading travel destinations data...")
    df = load_data(data_file)
    
    # Get user input for source city
    print("\n" + "="*50)
    print("INDIA TRAVEL RECOMMENDATION ENGINE")
    print("="*50)
    print("\nEnter 'exit' at any time to quit.")
    
    while True:
        source_city = input("\nEnter your source city: ").strip()
        
        if source_city.lower() == 'exit':
            print("\nThank you for using the India Travel Recommendation Engine!")
            break
            
        # Get recommendations
        recommendations = recommend_destinations(source_city, df)
        
        # Print results
        if not recommendations.empty:
            print_recommendations(recommendations, source_city)
        else:
            print(f"\nNo recommendations found for {source_city}. Please try another city.")
        
        # Show sample cities for next search
        sample_cities = df.sample(min(5, len(df)))['City'].tolist()
        print(f"\nTry these cities: {', '.join(sample_cities)}")
        print("Or type 'exit' to quit.")

if __name__ == "__main__":
    main()
