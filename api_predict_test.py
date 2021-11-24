import json

import pandas as pd
import requests

API_URL = 'http://localhost:8000'

model_key = 'C4O7ABW5VT6R25FZLBL8TDUH'


pred_records = pd.read_csv(
    'data/sample_multi_ts.csv').drop(columns=['numeric_0']).head().to_dict(orient='records')

r = requests.post(
    url=f'{API_URL}/predict/{model_key}',
    data=json.dumps({'data': pred_records})
)
print(r.json())
