# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Convert emails into numerical features using tokenization, lowercasing, and TF-IDF or Bag
 of Words.
 2. Transform the processed text into feature vectors for SVM input.
 3. Train an SVM classifier (with a linear or other kernel) on labeled data to distinguish
 between spam and not spam emails.
 4. Use the trained SVM model to predict whether new emails are spam and evaluate
 performance using metrics like accuracy and precision

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: sreeviveka V.S
RegisterNumber:  2305001031
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

df=pd.read_csv('/content/spamEX10.csv',encoding='ISO-8859-1')
df.head()

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,predictions))
print("Classification Report:")
print(classification_report(y_test,predictions))

def predict_message(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]

new_message="Free prixze money winner"
result=predict_message(new_message)
print(f"The message: '{new_message}' is classified as: {result}")
```

## Output:
![WhatsApp Image 2024-10-26 at 15 26 26_fea1c728](https://github.com/user-attachments/assets/971f0355-04b2-4932-b994-18c1bd5afe42)
![WhatsApp Image 2024-10-26 at 15 26 32_c45f7491](https://github.com/user-attachments/assets/375760b6-135f-4bf0-a6b3-6f4c5f2f8940)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
