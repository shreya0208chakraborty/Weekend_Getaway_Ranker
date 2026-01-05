# 🌍 India Weekend Getaway Recommender

## 🚀 Project Overview
A smart travel recommendation system that suggests personalized weekend getaway destinations across India based on your current city. The system uses geolocation, popularity metrics, and user ratings to recommend the best nearby destinations for short trips.

## ✨ Features
- **Smart Recommendations**: Suggests destinations based on multiple factors
- **Distance-Aware**: Considers travel distance from your location (up to 800km)
- **Popularity-Based**: Ranks places by user ratings and review counts
- **Category Filtering**: Includes various destination types (historical, religious, nature, etc.)
- **Customizable Search**: Adjustable parameters for distance and number of recommendations

## 🛠️ Tech Stack
- **Programming Language**: Python 3.8+
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical operations
  - `geopy` - Geographical distance calculations
  - `tqdm` - Progress bars for better UX

## 🏗️ Project Architecture
1. **Data Loading**: Loads and preprocesses the Indian travel destinations dataset
2. **Input Processing**: Takes user's current city as input
3. **Distance Calculation**: Computes distances using geodetic calculations
4. **Scoring**: Calculates a weighted score based on distance, rating, and popularity
5. **Ranking**: Sorts destinations by final score
6. **Output**: Displays top recommendations with relevant details

## 📊 Dataset
- **Source**: Curated dataset of top Indian tourist destinations
- **Size**: 300+ destinations across India
- **Key Attributes**:
  - City and State
  - Google review rating (1-5)
  - Number of reviews
  - Type of attraction
  - Location coordinates

## 🧠 Algorithm Explanation
The recommendation system uses a weighted scoring approach:

1. **Distance Score**: Normalized inverse distance (closer = better)
2. **Rating Score**: Normalized Google review ratings
3. **Popularity Score**: Based on number of reviews

Final score = (0.3 × Distance_Score) + (0.4 × Rating_Score) + (0.3 × Popularity_Score)

## 🎯 Sample Usage
```python
# Example usage
def recommend_destinations(
    source_city: str, 
    df: pd.DataFrame, 
    max_distance: float = MAX_DISTANCE_KM,
    top_n: int = 5
) -> pd.DataFrame:

### Example Output:
```
TOP 5 WEEKEND GETAWAYS FROM PUNE
====================================================================================================
City      State      Attraction       Distance (km) Category         Rating Popularity Match Score  
   Mumbai Maharastra     Marine Drive 120.1 km              Monument 4.5 ★  0.0K       76.7%        
   Mysore  Karnataka    Mysore Palace 750.7 km                Palace 4.6 ★  0.0K       46.1%        
Hyderabad  Telangana        Charminar 506.3 km           Film Studio 4.4 ★  0.0K       42.6%        
Ahmedabad    Gujarat Sabarmati Ashram 516.1 km            Historical 4.5 ★  0.0K       33.8%        
Bangalore  Karnataka Bangalore Palace 733.0 km      Botanical Garden 4.4 ★  0.0K       29.1%        
```

## 🚀 How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the recommendation engine:
   ```bash
   python src/recommend.py
   ```
4. Enter your current city when prompted

## � Project Structure
```
travel-recommendation-engine/
├── data/
│   └── Top_Indian_Places_to_Visit.csv  # Dataset
├── src/
│   └── recommend.py                    # Main recommendation logic
├── requirements.txt                    # Dependencies
└── README.md                           # Project documentation
```

## ⚠️ Limitations
- Limited to 50+ major Indian cities with predefined coordinates
- Doesn't account for real-time traffic or travel conditions
- Weather conditions not considered in recommendations
- Limited to weekend getaways within 800km

## � Future Improvements
- Add more cities and points of interest
- Include real-time weather data
- Integrate with travel APIs for real-time pricing
- Add user preferences and history
- Create a web/mobile interface
- Include travel time estimates using transportation modes

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
