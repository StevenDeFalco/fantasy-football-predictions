# Weekly NFL Fantasy Football Point Forecasting

**Machine Learning Pipeline for Predicting Weekly Fantasy Performance**

A comprehensive end-to-end machine learning project that predicts weekly NFL fantasy football points using gradient boosting models. Built on 8 years of play-by-play data (2009-2016) with rigorous time-aware evaluation.

## 📊 Project Overview

Fantasy football forecasting is challenging due to high week-to-week variance, context-dependent performance, and limited historical data per player. This project addresses these challenges by:

- Computing weekly fantasy points directly from NFL play-by-play data using Half-PPR scoring
- Engineering 51 time-aware features (lag variables, rolling averages, season-to-date aggregates)
- Training and comparing three regression models: Ridge, Random Forest, and Gradient Boosting
- Using strict time-based train/validation/test splits to prevent data leakage

## 🎯 Key Results

- **Best Model:** Histogram Gradient Boosting
- **Test MAE:** 2.62 fantasy points
- **R² Score:** 0.726 (72.6% variance explained)
- **Improvement:** 45% better than best baseline (5-week rolling average: 4.76 MAE)
- **Position-specific MAE:** WR 2.02 | RB 3.52 | QB 4.28

## 🗂️ Repository Structure

```
├── project.ipynb              # Main analysis notebook
├── report/
│   └── main.tex              # IEEE conference paper (LaTeX)
├── presentation/
│   ├── slides.tex            # Beamer presentation slides
│   └── script.tex            # Speaking script
├── data/
│   ├── fantasy_data.csv      # Processed fantasy points dataset
│   └── NFL Play by Play...  # Raw play-by-play data (2009-2018)
├── artifacts/
│   ├── models/               # Trained model files (.joblib)
│   ├── metrics/              # Performance metrics (CSV)
│   ├── predictions/          # Test set predictions
│   └── visuals/              # Generated plots (PNG)
└── IEEE Conference Latex Template/
```

## 🔬 Methodology

### Data Pipeline
1. **Play-by-Play → Player-Week Aggregates** → Aggregate individual plays to weekly player performance
2. **Feature Engineering** → Compute 51 temporal features (no current-week leakage)
3. **Time-Based Split** → Train (2009-2014), Validation (2015), Test (2016)
4. **Model Training & Evaluation** → Ridge, Random Forest, Gradient Boosting

### Feature Categories
- **Lag variables:** Previous week statistics
- **Rolling averages:** 3-week, 5-week, 8-week windows
- **Season-to-date aggregates:** Cumulative performance metrics
- **Trend indicators:** Recent performance vs. rolling averages
- **Usage metrics:** Touches, targets, attempts, red zone opportunities

## 📈 Key Findings

1. **Recent form dominates:** Rolling averages (3-week, 5-week) are the most predictive features
2. **Position variance:** Wide receivers are most predictable; quarterbacks show higher volatility
3. **Baseline improvement:** ML models substantially outperform naive heuristics (last week, rolling averages)
4. **Outlier challenge:** Models smooth toward expected values and struggle with extreme performances

## 🚀 Usage

### Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn
- joblib (for model persistence)

### Running the Analysis
```bash
# Open the main notebook
jupyter notebook project.ipynb

# Or run individual sections as needed
```

### Loading Trained Models
```python
import joblib
model = joblib.load('artifacts/models/best_model_gb.joblib')
```

## 📄 Deliverables

- **[project.ipynb](project.ipynb)** - Complete analysis workflow with 80+ cells
- **[report/main.tex](report/main.tex)** - IEEE conference paper format
- **[presentation/slides.pdf](presentation/slides.pdf)** - 8-slide Beamer presentation
- **[presentation/script.pdf](presentation/script.pdf)** - Aligned speaking script (~6-7 minutes)

## 🎓 Academic Context

**Course:** AAI 595 Applied Machine Learning  
**Institution:** Stevens Institute of Technology  
**Authors:** Steven DeFalco, Liam Guske, Nick Obiso  
**Semester:** Fall 2025

## 📊 Dataset Attribution

NFL Play-by-Play data sourced from [Kaggle](https://www.kaggle.com/datasets) (2009-2018 seasons). Analysis focused on 2009-2016 regular season data for consistency.

## 🔮 Future Work

- Incorporate matchup-specific features (opponent defense rankings)
- Integrate injury and weather data
- Ensemble predictions combining multiple models
- Multi-step forecasting (predict next 2-3 weeks)
- Real-time weekly prediction dashboard

## 📝 License

This project is for academic purposes as part of AAI 595 coursework.

---

**Note:** All reported metrics are from the held-out 2016 test set using strict temporal validation to ensure realistic forecasting conditions.
