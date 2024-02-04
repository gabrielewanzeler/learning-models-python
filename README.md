Diabetes Prediction Project


Overview
This Python project focuses on predicting the likelihood of diabetes in individuals using machine learning models. The dataset used for training and testing contains various health-related features such as age, BMI, HbA1c levels, blood glucose levels, gender, hypertension, smoking history, heart disease, and the target variable, diabetes.

Technologies and Libraries Used
Pandas and NumPy: For data manipulation and handling.
Seaborn and Matplotlib: For data visualization and plotting graphs.
Scikit-learn: For machine learning tasks, including model creation, training, and evaluation.
Keras: Used for building and training a neural network.
MinMaxScaler: For normalizing numeric features.
RandomForestClassifier, KNeighborsClassifier, DecisionTreeClassifier: Various classifiers for model comparison.
Data Preprocessing
The dataset undergoes several preprocessing steps, including handling missing or incorrect values, removing duplicates, and normalizing numeric columns. Categorical columns are encoded using LabelEncoder.

Exploratory Data Analysis (EDA)
EDA is performed using Seaborn and Matplotlib to gain insights into the dataset, visualizing the distribution of gender, diabetes prevalence, and age distribution of diabetic individuals.

Machine Learning Models
The project explores the following machine learning models:

K-Nearest Neighbors (KNN)
Decision Tree Classifier
Random Forest Classifier
Neural Network (using Keras)
Each model is trained, tested, and evaluated, and their performances are visualized through confusion matrices, ROC curves, and accuracy scores.

Results and Comparisons
The final section presents a graphical representation of the accuracy achieved by each model, allowing for a quick comparison. The accuracy scores provide insights into how well each model performs in predicting diabetes based on the given features.
