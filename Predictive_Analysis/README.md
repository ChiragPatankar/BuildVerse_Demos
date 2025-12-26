# 📊 Predictive Analytics Dashboard

A professional, interactive machine learning dashboard built with Streamlit for real-time predictive analytics.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Live Demo

**[View Live App →](https://buildverse-predictive-dashboard.streamlit.app)**

## ✨ Features

### 📈 Data Explorer
- Interactive data visualization
- Statistical summaries
- Distribution analysis
- Correlation matrices
- Support for CSV uploads

### 🤖 Model Training
- **Multiple Algorithms:**
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Linear Regression
- Real-time training and evaluation
- Performance metrics (R², RMSE, MAE)
- Feature importance analysis
- Actual vs Predicted visualizations

### 🎯 Predictions
- Interactive input forms
- Real-time predictions
- Beautiful result displays
- Input validation

### 📊 Reports & Export
- Downloadable performance reports
- CSV export of predictions
- Comprehensive model summaries

## 🛠️ Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/predictive-analytics-dashboard.git
cd predictive-analytics-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Docker (Optional)

```bash
docker build -t predictive-dashboard .
docker run -p 8501:8501 predictive-dashboard
```

## 📦 Deployment

### Streamlit Community Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

Your app will be live at: `https://your-app-name.streamlit.app`

### Other Platforms

- **Heroku:** Use `setup.sh` and `Procfile`
- **AWS/Azure/GCP:** Deploy as containerized app
- **Railway/Render:** Direct GitHub integration

## 📊 Sample Datasets

The app includes three built-in datasets:

1. **Sales Forecast** - Predict sales based on marketing metrics
2. **Housing Prices** - Estimate property values
3. **Customer Churn** - Analyze customer retention

You can also upload your own CSV files!

## 🎯 Usage

### Basic Workflow

1. **Select Data Source**
   - Choose a sample dataset or upload your own CSV
   
2. **Configure Model**
   - Select algorithm (Random Forest, Gradient Boosting, Linear Regression)
   - Adjust hyperparameters
   - Set train/test split
   
3. **Train Model**
   - Click "Train Model" button
   - View performance metrics
   - Analyze feature importance
   
4. **Make Predictions**
   - Enter feature values
   - Get instant predictions
   - Export results

### Custom Dataset Requirements

Your CSV should have:
- Numeric features (categorical features will be encoded automatically)
- One target variable (continuous for regression)
- No missing critical values

## 🔧 Configuration

### Model Parameters

**Random Forest:**
- `n_estimators`: Number of trees (10-200)
- `max_depth`: Maximum tree depth (3-20)

**Gradient Boosting:**
- `n_estimators`: Number of boosting stages (10-200)
- `max_depth`: Maximum tree depth (3-20)

**Linear Regression:**
- No hyperparameters required

### Advanced Settings

Edit `app.py` to customize:
- Color schemes
- Chart styles
- Default parameters
- Additional algorithms

## 📈 Performance Metrics

The dashboard calculates:

- **R² Score:** Coefficient of determination
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **Feature Importance:** For tree-based models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- ML models powered by [scikit-learn](https://scikit-learn.org/)
- Visualizations using [Plotly](https://plotly.com/)



---

⭐ Star this repo if you find it helpful!

**Made with ❤️ and Streamlit**
