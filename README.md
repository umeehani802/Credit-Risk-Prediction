# 💳 Credit Risk Prediction
A machine learning project to predict credit risk using logistic regression and decision trees.

## 🎯 Objective

Predict whether a loan applicant is likely to default on a loan using machine learning techniques.

## 🧠 Skills Practiced

- ✅ Data cleaning and handling missing values  
- 📊 Exploratory Data Analysis (EDA)  
- 🤖 Binary classification using machine learning (Logistic Regression, Decision Tree)  
- 📈 Model evaluation using confusion matrix, accuracy, and classification report  

## 📂 Dataset

The dataset includes features such as:
- Age, Income, Loan Amount, Credit Score
- Employment details, Marital status, Education level
- Debt-to-Income Ratio, Interest Rate, Loan Term
- And more...

The final dataset used for training contains **255,347** rows with **no missing values**.

---

## 🔍 Approach

1. Loaded and explored the dataset  
2. Performed basic cleaning (label encoding, feature scaling)  
3. Visualized patterns and outliers using Seaborn and Matplotlib  
4. Split data into training and testing sets  
5. Applied classification models:
   - Logistic Regression
   - Decision Tree Classifier  
6. Evaluated model performance using confusion matrix and classification report

---

## 📊 Evaluation Results

```python
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

✅ Accuracy
0.885

📊 Confusion Matrix
[[45007   132]
 [ 5739   192]]

📋 Classification Report
              precision    recall  f1-score   support

           0       0.89      1.00      0.94     45139  
           1       0.59      0.03      0.06      5931  

    accuracy                           0.89     51070  
   macro avg       0.74      0.51      0.50     51070  
weighted avg       0.85      0.89      0.84     51070  

📈 Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

🔎 Insights
The model achieves high accuracy by predicting non-defaulters correctly.
However, recall for defaulters is very low.
Indicates a class imbalance problem that should be addressed in future versions.

🚀 Future Work
Apply SMOTE or other resampling techniques to balance classes
Try ensemble models (Random Forest, XGBoost)
Tune hyperparameters to improve recall on minority class
Use ROC-AUC for more balanced evaluation metrics

🧾 License
This project is for educational and academic use only.







