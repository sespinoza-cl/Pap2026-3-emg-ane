import requests, json

TOKEN = 'I6fw7YG3Nd1zvc6Y8ao4cYkiZibygdqnYA3P2FExTX3qhEuisgzwL1SwerVV'
RECORD_ID = '20601653'
headers = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}

# Fetch current metadata to preserve existing fields
r = requests.get(f'https://zenodo.org/api/deposit/depositions/{RECORD_ID}',
                 headers={'Authorization': f'Bearer {TOKEN}'})
current = r.json()['metadata']

updated = {
    "metadata": {
        # Keep existing
        "title": current["title"],
        "description": current["description"],
        "license": current.get("license", "cc-by-4.0"),
        "access_right": current.get("access_right", "open"),
        "publication_date": current.get("publication_date", "2026-06-09"),

        # Fix: resource type (Zenodo API v1 uses upload_type)
        "upload_type": "dataset",

        # Solo el curador del dataset
        "creators": [
            {
                "name": "Espinoza, Sebastian",
                "affiliation": "Andres Bello National University; Universidad de Valparaiso",
                "orcid": "0000-0001-9678-2665"
            }
        ],

        # Fix: keywords
        "keywords": [
            "electromyography", "masseter", "mastication", "topical anesthesia",
            "lidocaine", "sensorimotor compensation", "central pattern generator",
            "BIDS", "EMG", "somatosensory feedback", "oral neuroscience"
        ],

        # Fix: related identifiers
        "related_identifiers": [
            {
                "relation": "isSupplementTo",
                "identifier": "https://github.com/sespinoza-cl/Pap2026-3-emg-ane",
                "resource_type": "software",
                "scheme": "url"
            }
        ]
    }
}

r = requests.put(
    f'https://zenodo.org/api/deposit/depositions/{RECORD_ID}',
    headers=headers,
    data=json.dumps(updated)
)

print('Status:', r.status_code)
if r.status_code == 200:
    m = r.json()['metadata']
    print('resource_type:', m.get('resource_type'))
    print('creators:', len(m.get('creators', [])), 'autores')
    print('keywords:', len(m.get('keywords', [])), 'keywords')
    print('related_ids:', len(m.get('related_identifiers', [])), 'links')
    print('\nTodo OK — listo para Publish')
else:
    print('Error:', r.text[:500])
