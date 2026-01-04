# CrediSense- A Credit Risk Prediction Application using Machine Learning
CrediSense is an intelligent credit risk prediction system that leverages machine learning algorithms to assess the creditworthiness of loan applicants. Built with Streamlit, the application provides an intuitive interface for financial institutions to make data-driven lending decisions while minimizing default risks. <br>
🔗 **Live Demo:** https://credisense.streamlit.app

## Key Features

- Multi-Model Ensemble: Implements and compares four powerful machine learning algorithms
- Interactive Dashboard: Real-time predictions through an intuitive Streamlit interface
- Comprehensive Evaluation: Multiple performance metrics for robust model assessment
- Data Visualization: Clear insights into model performance and feature importance
- Production-Ready: Optimized for deployment in real-world financial environments

## Models Implemented
- Decision Trees: Baseline tree-based classifier with interpretable decision rules.
- Random Forest: Ensemble method using bootstrap aggregation and random feature selection to reduce overfitting.
- Extra Trees: Variant of Random Forest with additional randomization in split selection for better generalization.
- XGBoost: Gradient boosting framework with regularization, typically achieving best performance on structured data.

The performance of all implemented models was evaluated using Accuracy, F1-Score, and AUC-ROC to ensure a balanced and reliable assessment.

| Model          | Accuracy | F1-Score | AUC-ROC |
|----------------|----------|----------|---------|
| Extra Trees    | 0.6476   | 0.6838   | 0.6912  |
| Random Forest  | 0.6190   | 0.6667   | 0.6735  |
| XGBoost        | 0.6286   | 0.6549   | 0.6455  |
| Decision Tree  | 0.5810   | 0.6207   | 0.5693  |

**Best Model Selected:** Extra Trees Classifier  
The Extra Trees model achieved the highest AUC-ROC score, indicating superior class separation capability and overall robustness for credit risk prediction.
 
## Tech Stack
- **Python 3.8+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit

## Dataset Used

**German Credit Dataset** - 1,000 credit applications with 20 features including demographic attributes, financial indicators, and loan characteristics. Binary target variable indicates creditworthiness. 
<a href="https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk">Link to the dataset</a> <br>

Note: The dataset is not included in this repository due to Kaggle’s licensing restrictions.
Please ensure the dataset is downloaded manually and placed in the correct directory before running the project.

## Pipeline Workflow

1. **Data Ingestion**: Load German Credit Dataset
2. **EDA**: Feature distributions, correlations, class imbalance analysis
3. **Preprocessing**: Handle missing values, encode categorical variables, scale numerical features
4. **Model Training**: Train Decision Trees, Random Forest, Extra Trees, and XGBoost
5. **Evaluation**: Compare models using Accuracy, F1-Score, AUC-ROC, and Confusion Matrices
6. **Deployment**: Integrate best model into Streamlit application

## Installation

```bash
# Clone repository
git clone https://github.com/Madhuri36/CrediSense.git
cd CrediSense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup
1. Visit the link above and download the dataset (german_credit_data.csv).
2. Extract the file if it is in a ZIP format.
3. Create a folder named data in the project root (if not already present).
4. Place the downloaded CSV file inside the data/ directory.

### Running Web Application

```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`
