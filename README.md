# Fashion Forward Forecasting: ML Product Recommendation Prediction
The project builds a machine learning pipeline to predict whether customers would recommend a product based on their reviews, age, and product information. This was built for StyleSense, an online women's clothing retailer, to help handle the backlog of product reviews with missing recommendation data.

## Getting Started
Instructions for how to get a copy of the project running on your local machine.

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- spacy
- xgboost
- matplotlib
- seaborn

## Installation
1. Clone the repository:
    git clone https://github.com/NarrativeWeaver/dsnd-pipelines-project.git
   cd dsnd-pipelines-project
   
2. Install required packages:
    python -m pip install -r requirements.txt
   
3. Download the spaCy model:
    python -m spacy download en_core_web_sm
   
4. Open the Jupyter notebook:
   jupyter notebook starter.ipy


## Testing
The model pipeline can be tested by running all cells in the notebook, which includes evaluation metrics on the test dataset.

## Project Instructions

### Data Exploration:
- Analyzed class distribution revealing an imbalance (18% not recommended, 82% recommended)
- Explored distributions of age, positive feedback counts, and department categories
- Examined text patterns between recommended and non-recommended products
- Created visualizations to understand feature relationships

### Pipeline Construction:
- Built a comprehensive pipeline with specialized components for:
   - Numeric features (Age, Positive Feedback Count)
   - Categorical features (Clothing ID, Division, Department, Class Name)
   - Text features (Review Title and Text)
- Implemented custom text processing using spaCy with lemmatization
- Created TF-IDF vectorization with n-gram features

### Model Training
- Trained and compared Random Forest and XGBoost models
- Implemented class-balanced weighting to address imbalanced classes
- Achieved initial accuracies of approximately 84% before tuning

### Hyperparameter Tuning
- Used GridSearchCV to find optimal parameters for both models
- Improved model performance by 1-2% after tuning
- Identified the most important feature groups (text, categorical, numerical)

### Model Evaluation
- Evaluated models using accuracy, precision, recall, and F1-score
- Generated confusion matrices to visualize classification performance
- Analyzed feature importance to understand driving factors in predictions
- Final model achieved approximately 86% accuracy on the test set

## Built With
- pandas - Data manipulation and analysis
- scikit-learn - Pipeline construction and model training
- spaCy - Advanced natural language processing
- XGBoost - Gradient boosting implementation
- matplotlib/seaborn - Data visualization
- NumPy - Numerical computations

## License
This project is licensed under the MIT License 
