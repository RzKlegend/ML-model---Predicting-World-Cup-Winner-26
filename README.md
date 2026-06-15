# World Cup Winner Predictor - Random Forest Model

A machine learning solution that uses historical international football match data (1872-2025) to predict the 2026 FIFA World Cup winner using a Random Forest classifier.

## Dataset Information

**Source**: [International Football Results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

The dataset contains historical match results including:
- Home/Away team names
- Match outcomes (win, draw, loss)
- Goal differences and scores
- Historical performance metrics for each nation

## Installation Requirements

```bash
pip install pandas numpy scikit-learn requests
```

## How to Run

### Option 1: Using Kaggle API (Recommended)

1. **Get your Kaggle API token:**
   - Go to [Kaggle Settings](https://www.kaggle.com/settings/api)
   - Generate an API key if you don't have one
   
2. **Set environment variable:**
   
   ```bash
   # Windows PowerShell
   $env:KAGGLE_API_TOKEN="your_api_token_here"
   
   # Or create a .env file with your Kaggle credentials
   pip install python-dotenv
   export KAGGLE_CONFIG_DIR=~/.config/kaggle  # Linux/Mac
   
3. **Run the application:**
   ```bash
   python app.py
   ```

### Option 2: Using Local CSV File

1. Download the dataset from Kaggle manually (right-click → "Download")
2. Save it as `football_data.csv` in the same directory
3. Update line 78 in `app.py`:
   ```python
   self.data = pd.read_csv('path/to/your/football_data.csv')
   ```

### Option 3: Using Sample Data (For Testing)

The application includes a fallback that generates sample data when real data is unavailable, allowing you to test the model structure.

## Model Architecture

```
FootballPredictor Class
├── Random Forest Classifier (trained on historical match outcomes)
└── Team Statistics Calculator
   
Features Used:
- Goals For/Against ratio
- Goal Difference trends
- Home/Away performance metrics
- Historical win percentages
```

## Output Format

The model outputs a prediction dictionary containing:
- **Winner**: Predicted 2026 World Cup champion
- **Confidence Level**: Model confidence percentage (75.3%)
- **Top Contenders**: List of teams with highest probability to win

Example output:
```
Predicted Winner: Brazil
Confidence Level: 75.3%

Top Contenders:
  - Argentina
  - France  
  - England
```

## Customization Options

### Modify Feature Engineering (Line ~108)
Edit the `prepare_features()` method to include additional features like:
- Team age/formation data
- Player statistics
- Head-to-head records
- Recent form analysis

### Adjust Model Parameters (Line 34)
```python
self.model = RandomForestClassifier(
    n_estimators=100,      # Number of trees in forest
    max_depth=None,        # No limit on tree depth
    random_state=42,       # Reproducibility
    class_weight='balanced' # Handle imbalanced classes
)
```

## License

This project uses data from Kaggle under their terms of service. Please review the dataset license for usage restrictions.

---

**Note**: This is a baseline Random Forest model demonstrating the approach to World Cup prediction using historical football match data. For production use, consider implementing more sophisticated features and validation techniques.
