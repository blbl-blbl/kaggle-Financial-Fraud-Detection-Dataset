# kaggle-Financial-Fraud-Detection-Dataset

**[Link to Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset/data)**


Ideas for Feature engineering:
  * df['log_amount'] = np.log1p(df['amount'])
  * df['hour'] = (df['step'] - 1) % 24
  * df['day'] = (df['step'] - 1) // 24
  * df['dest_type'] = df['nameDest'].str[0]