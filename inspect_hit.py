import json
from app.typesense_engine import get_typesense_client
from app import config

client = get_typesense_client()
res = client.collections[config.TYPESENSE_COLLECTION].documents.search({
    'q': 'account',
    'query_by': 'question,answer',
    'per_page': 2
})

print(json.dumps(res, indent=2))
