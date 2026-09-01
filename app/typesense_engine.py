import logging
import typesense
from typing import Any, Dict, List, Optional
from app import config

logger = logging.getLogger("typesense_engine")


def get_typesense_client() -> typesense.Client:
    """Initialize and return the Typesense client."""
    return typesense.Client({
        'nodes': [{
            'host': config.TYPESENSE_HOST,
            'port': str(config.TYPESENSE_PORT),
            'protocol': config.TYPESENSE_PROTOCOL,
        }],
        'api_key': config.TYPESENSE_API_KEY,
        'connection_timeout_seconds': 5,
    })


def ensure_faq_collection(client: typesense.Client) -> None:
    """Ensure the FAQ hybrid collection exists with the 768-d vector field."""
    schema = {
        'name': config.TYPESENSE_COLLECTION,
        'fields': [
            {'name': 'id', 'type': 'string'},
            {'name': 'workspace_id', 'type': 'int32', 'facet': True},
            {'name': 'question', 'type': 'string'},
            {'name': 'answer', 'type': 'string'},
            {'name': 'priority', 'type': 'int32'},
            {'name': 'is_active', 'type': 'bool'},
            {'name': 'embedding', 'type': 'float[]', 'num_dim': 768},
            {'name': 'lexicon_terms', 'type': 'string[]', 'optional': True},
            {'name': 'document_type', 'type': 'string', 'optional': True, 'facet': True},
        ],
        'default_sorting_field': 'priority',
    }

    try:
        col = client.collections[config.TYPESENSE_COLLECTION].retrieve()
        logger.info("Typesense collection '%s' already exists.", config.TYPESENSE_COLLECTION)
        fields = [f['name'] for f in col.get('fields', [])]
        update_fields = []
        if 'lexicon_terms' not in fields:
            logger.info("Adding 'lexicon_terms' field to Typesense collection '%s'...", config.TYPESENSE_COLLECTION)
            update_fields.append({'name': 'lexicon_terms', 'type': 'string[]', 'optional': True})
        if 'document_type' not in fields:
            logger.info("Adding 'document_type' field to Typesense collection '%s'...", config.TYPESENSE_COLLECTION)
            update_fields.append({'name': 'document_type', 'type': 'string', 'optional': True, 'facet': True})
        if update_fields:
            client.collections[config.TYPESENSE_COLLECTION].update({'fields': update_fields})
    except typesense.exceptions.ObjectNotFound:
        logger.info("Creating Typesense collection '%s'...", config.TYPESENSE_COLLECTION)
        client.collections.create(schema)
        logger.info("Collection '%s' created successfully.", config.TYPESENSE_COLLECTION)
    except Exception as e:
        logger.warning("Typesense connection/schema check warning: %s", e)


def upsert_faq_document(client: typesense.Client, document: Dict[str, Any]) -> Dict[str, Any]:
    """Upsert an FAQ document with its vector embedding into Typesense."""
    return client.collections[config.TYPESENSE_COLLECTION].documents.upsert(document)


def delete_faq_document(client: typesense.Client, doc_id: str) -> Dict[str, Any]:
    """Delete an FAQ document by ID from Typesense."""
    try:
        return client.collections[config.TYPESENSE_COLLECTION].documents[str(doc_id)].delete()
    except typesense.exceptions.ObjectNotFound:
        return {'id': str(doc_id), 'deleted': False}


def execute_hybrid_search(
    client: typesense.Client,
    query_text: str,
    query_vector: List[float],
    workspace_id: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Execute native Typesense hybrid search using multi_search (POST) to support large 768-d vector payloads."""
    vector_str = f"embedding:([{','.join(f'{x:.5f}' for x in query_vector)}], k:{top_k})"

    search_query = {
        'collection': config.TYPESENSE_COLLECTION,
        'q': query_text if query_text.strip() else '*',
        'query_by': 'question,answer,lexicon_terms',
        'query_by_weights': '4,2,3',
        'vector_query': vector_str,
        'per_page': top_k,
    }

    filter_clauses = ['is_active:=true']
    if workspace_id is not None:
        filter_clauses.append(f'workspace_id:={workspace_id}')

    search_query['filter_by'] = ' && '.join(filter_clauses)

    response = client.multi_search.perform({'searches': [search_query]}, {})
    if response and 'results' in response and len(response['results']) > 0:
        return response['results'][0]
    return {'hits': []}

