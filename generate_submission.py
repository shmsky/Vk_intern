import joblib
import pandas as pd
import numpy as np

test_df = pd.read_csv('test.parquet')

def create_features(data):
    if 'label' in data:
      df = pd.DataFrame({
          'id': data['id'],
          'dates': data['dates'],
          'values': data['values'],
          'label': data['label']
      })
    else:
      df = pd.DataFrame({
          'id': data['id'],
          'dates': data['dates'],
          'values': data['values'],
      })

    # Генерация признаков
    df['mean'] = df['values'].apply(np.mean)
    df['std'] = df['values'].apply(np.std)
    df['min'] = df['values'].apply(np.min)
    df['max'] = df['values'].apply(np.max)
    df['median'] = df['values'].apply(np.median)

    # Скользящие средние
    df['rolling_mean'] = df['values'].apply(lambda x: pd.Series(x).rolling(window=7).mean().iloc[-1])
    df['rolling_std'] = df['values'].apply(lambda x: pd.Series(x).rolling(window=7).std().iloc[-1])

    return df

test_features = create_features(test_df)

model = joblib.load('best_model.joblib')

y_test_pred = model.predict_proba(test_features)[:, 1]

# Формирование результата
submission = pd.DataFrame({
    'id': test_df['id'],
    'score': y_test_pred
})

# Сохранение результата
submission.to_csv('sample_submission.csv', index=False)