import requests
from pathlib import Path

TOKEN = 'I6fw7YG3Nd1zvc6Y8ao4cYkiZibygdqnYA3P2FExTX3qhEuisgzwL1SwerVV'
RECORD_ID = '20601653'
BIDS = Path(r'C:\Users\Pc - Casa\Desktop\Proyectos_Claude\Phd\Paper3\zenodo_emg_bids')
MISSING = ['M6', 'PS4', 'PS7', 'PS8', 'PS14', 'S7', 'S8', 'S11']

headers = {'Authorization': f'Bearer {TOKEN}'}
dep = requests.get(
    f'https://zenodo.org/api/deposit/depositions/{RECORD_ID}',
    headers=headers
).json()
bucket = dep['links']['bucket']

for s in MISSING:
    fpath = BIDS / f'sub-{s}' / 'emg' / f'sub-{s}_task-chewing_events.tsv'
    fname = f'sub-{s}_task-chewing_events.tsv'
    print(f'Subiendo {fname}...', end=' ')
    with open(fpath, 'rb') as f:
        r = requests.put(f'{bucket}/{fname}', data=f, headers=headers)
    print(r.status_code)

print('Listo.')
