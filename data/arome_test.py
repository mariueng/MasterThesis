import requests
import pandas as pd

client_id = 'f2acda91-356a-4475-b815-17214a0c7f14'
endpoint = "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=60.10&lon=9.58"
r = requests.get(endpoint, headers={'Authorization': client_id})
print(r.status_code)
print(r.url)
json = r.json()
data = None
if r.status_code == 200:
    data = json['data']
else:
    print('Error! Returned status code %s' % r.status_code)
    print('Message: %s' % json['error']['message'])
    print('Reason: %s' % json['error']['reason'])
df = pd.DataFrame()
for i in range(len(data)):
    print(data[i])
    assert False
    df = df.append(row)
df = df.reset_index()

