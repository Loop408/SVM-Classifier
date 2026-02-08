# Support Vector Machine (SVM) Classifier

An end-to-end machine learning application for building and training Support Vector Machine classifiers with an interactive web interface.

## Features

- **Data Ingestion**: Download the Iris dataset directly or upload your own CSV files
- **Exploratory Data Analysis (EDA)**: Visualize data distributions and correlations
- **Data Cleaning**: Handle missing values and prepare data for modeling
- **Model Training**: Train SVM classifiers with customizable hyperparameters
- **Interactive Dashboard**: Built with Streamlit for easy interaction and visualization
- **Model Evaluation**: View metrics like accuracy and confusion matrices

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The `data/raw` and `data/clean` folders will be created automatically when you run the application for the first time. No manual folder creation is required.

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Workflow
1. **Choose a Data Source**: Download the Iris dataset or upload your own CSV
2. **Explore the Data**: View statistics, distributions, and correlations
3. **Configure Model Settings**: Select kernel type, regularization parameter (C), and gamma
4. **Clean the Data**: Handle missing values and prepare features
5. **Train & Evaluate**: Train the SVM model and view performance metrics

## Project Structure

```
SVM/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
└── data/
    ├── raw/              # Raw input datasets
    ├── clean/            # Cleaned processed datasets
    └── cleaned/          # Additional cleaned data storage
```

## Model Settings

- **Kernel**: Choose between linear, RBF, polynomial, or sigmoid kernels
- **Regularization (C)**: Control the trade-off between margin and classification error (0.01 - 10.0)
- **Gamma**: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (scale or auto)

## Dependencies

- **streamlit**: Web framework for data applications
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **scikit-learn**: Machine learning algorithms
- **seaborn**: Statistical data visualization

## License

This project is open source and available under the MIT License.
