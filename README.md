# banking_churn
This project predicts bank customer churn using features like demographics, account details, and activity. Data was analyzed, preprocessed, and modeled with Decision Tree, SVC, AdaBoost, Logistic Regression, and Random Forest. Model accuracies were compared to identify patterns and help the bank target at-risk customers for retention.


:this with more details


🏦 Banking Churn Analysis & Modeling
📌 Project Overview

This project focuses on analyzing customer churn in a banking dataset and building predictive machine learning models to identify customers who are likely to leave the bank.
The goal is to help the bank take proactive actions to retain valuable customers and improve profitability.

🎯 Business Objective

Identify customers most likely to churn.

Understand the key factors influencing their decision.

Apply predictive analytics to reduce churn rate.

💡 Business Value

Customer retention is critical for a bank’s success. By predicting churn, the bank can:

Offer personalized services to high-risk customers.

Increase retention rates.

Boost profitability.

📊 Data Exploration

The dataset contains 10,000 customer records with demographic, account, and activity details.

Sample Features:

CreditScore: Customer’s credit score.

Geography: Country (France, Germany, Spain).

Gender: Male or Female.

Age: Customer’s age.

Tenure: Years with the bank.

Balance: Account balance.

NumOfProducts: Products owned.

HasCrCard: Credit card ownership.

IsActiveMember: Activity status.

EstimatedSalary: Annual estimated salary.

Exited: Churn status (Target variable).

📈 Exploratory Data Analysis (EDA)

Geography Distribution: Most customers are from France (~50.1%).

Gender Distribution: 54.6% Male, 45.4% Female.

Churn Rate: ~20.4% customers exited.

Credit Score Distribution: Most customers have fair to good credit scores.

Salary & Balance: Wide distribution, with some outliers.

Age: Older customers have slightly higher churn rates.

🛠 Machine Learning Models

Models were trained and evaluated on churn prediction:

Logistic Regression

Train Accuracy: ~81.1%

Test Accuracy: ~81.1%

Metrics: Accuracy, Precision, Recall, Confusion Matrix.

(Additional models like Random Forest, XGBoost, SVM, etc. can be included if trained.)

📂 Project Structure
├── data-science-end-to-end-project.ipynb  # Main analysis notebook
├── dataset.csv                            # Banking dataset (if available)
├── README.md                              # Project documentation

🚀 How to Run

Clone the repository.

Install dependencies:

pip install -r requirements.txt


Open Jupyter Notebook:

jupyter notebook data-science-end-to-end-project.ipynb

📌 Key Insights

Geography, age, and activity status are major churn indicators.

Targeted retention strategies for specific customer segments can reduce churn.

Do you want me to also add the first 5 models’ descriptions in this GitHub README so it becomes complete? That way it matches your request for "الخليتني لأول 5 موديل".

You said:
write it in oe paragraph
ChatGPT said:

This project analyzes banking customer churn using a dataset of 10,000 records containing demographic, account, and behavioral features such as credit score, geography, age, tenure, balance, number of products, and activity status, with the target variable indicating whether a customer exited the bank. After performing exploratory data analysis to understand churn patterns, key trends were identified, including higher churn rates among certain geographies, older customers, and inactive members. The data was preprocessed and used to train multiple machine learning models—including Decision Tree, SVC, AdaBoost, Logistic Regression, and Random Forest—to predict churn. Each model’s training and testing accuracy was evaluated, revealing valuable insights that can help the bank target at-risk customers with retention strategies, thereby improving customer satisfaction and profitability.

You said:
less than 350 char
ChatGPT said:

This project predicts bank customer churn using features like demographics, account details, and activity. Data was analyzed, preprocessed, and modeled with Decision Tree, SVC, AdaBoost, Logistic Regression, and Random Forest. Model accuracies were compared to identify patterns and help the bank target at-risk customers for retention.

You said:
write read me file
ChatGPT said:

Here’s a concise README.md for your project:

Bank Customer Churn Prediction
📌 Project Overview

This project predicts whether a bank customer will churn based on demographic, account, and activity-related features. The goal is to identify at-risk customers and help the bank take proactive retention measures.

📊 Dataset

Records: 10,000 customers

Features: Credit score, geography, age, tenure, balance, number of products, credit card ownership, activity status, estimated salary

Target: Exited (1 = churned, 0 = stayed)

⚙️ Workflow

Exploratory Data Analysis (EDA)

Studied churn distribution, demographics, and account behaviors.

Preprocessing

Handled missing values, scaled numerical features, encoded categorical data.

Model Training

Models used:

Decision Tree Classifier

Support Vector Classifier (SVC)

AdaBoost Classifier

Logistic Regression

Random Forest Classifier

Evaluation

Compared training & testing accuracy for each model.

📈 Results

The models were evaluated based on accuracy.

Insights help in targeting customers likely to churn.

🚀 Installation
pip install -r requirements.txt

▶️ Usage
python churn_prediction.py

📌 Future Work

Hyperparameter tuning

Feature importance analysis

Deploying model with FastAPI
