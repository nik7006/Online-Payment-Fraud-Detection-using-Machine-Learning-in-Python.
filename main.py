# Install required library
!pip install xgboost

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay

from xgboost import XGBClassifier
from google.colab import files

# Upload Dataset
uploaded = files.upload()

# Load Dataset
df = pd.read_csv("mock_onlinefraud.csv")

print("Dataset Info")
print(df.info())

# Data Cleaning
if 'isFlaggedFraud' in df.columns:
    df.drop('isFlaggedFraud', axis=1, inplace=True)

# Encode Transaction Type
df['type'] = df['type'].map({
    'PAYMENT':0,
    'CASH_IN':1,
    'DEBIT':2,
    'CASH_OUT':3,
    'TRANSFER':4
})

# Remove unnecessary columns
df.drop(['nameOrig','nameDest','newbalanceOrig','newbalanceDest'], axis=1, inplace=True)

# Split features and target
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
lr = LogisticRegression(max_iter=1000,class_weight='balanced')
gb = GradientBoostingClassifier()
xgb = XGBClassifier(use_label_encoder=False,eval_metric='logloss')
svm = SVC(kernel='rbf',probability=True,class_weight='balanced')
knn = KNeighborsClassifier(n_neighbors=5)

models = {
    "Random Forest":rf,
    "Logistic Regression":lr,
    "Gradient Boosting":gb,
    "XGBoost":xgb,
    "SVM":svm,
    "KNN":knn
}

roc_data = {}

# Training and Evaluation
for name,model in models.items():
    
    model.fit(X_train,y_train)
    
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]
    
    print("\n==============================")
    print(name)
    print("==============================")
    
    print(classification_report(y_test,pred))
    
    cm = confusion_matrix(y_test,pred)
    
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=["Non-Fraud","Fraud"]).plot()
    
    plt.title(name + " Confusion Matrix")
    plt.show()
    
    fpr,tpr,_ = roc_curve(y_test,prob)
    roc_auc = auc(fpr,tpr)
    
    roc_data[name] = (fpr,tpr,roc_auc)

# ROC Curve Comparison
plt.figure(figsize=(8,6))

for name,(fpr,tpr,roc_auc) in roc_data.items():
    plt.plot(fpr,tpr,label=f"{name} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - All Models")
plt.legend()
plt.grid(True)
plt.show()
