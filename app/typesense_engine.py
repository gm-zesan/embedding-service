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


def delete_faq_document(
    client: typesense.Client,
    doc_id: str,
    workspace_id: int,
) -> Dict[str, Any]:
    """Delete an FAQ document by ID from Typesense, strictly scoped to workspace_id."""
    if not isinstance(workspace_id, int) or isinstance(workspace_id, bool) or workspace_id <= 0:
        raise ValueError(f"workspace_id must be a positive integer, got: {workspace_id!r}")
    clean_id = str(doc_id).strip() if doc_id is not None else ""
    if not clean_id:
        raise ValueError(f"doc_id must be a non-empty string, got: {doc_id!r}")

    try:
        return client.collections[config.TYPESENSE_COLLECTION].documents.delete({
            'filter_by': f'id:={clean_id} && workspace_id:={workspace_id}'
        })
    except typesense.exceptions.ObjectNotFound:
        return {'id': clean_id, 'deleted': False, 'num_deleted': 0}


def execute_hybrid_search(
    client: typesense.Client,
    query_text: str,
    query_vector: Optional[List[float]] = None,
    workspace_id: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Execute Typesense search: hybrid vector+text if query_vector is provided, or pure BM25 keyword search if query_vector is None."""
    search_query = {
        'collection': config.TYPESENSE_COLLECTION,
        'q': query_text if query_text.strip() else '*',
        'query_by': 'question,answer,lexicon_terms',
        'query_by_weights': '4,2,3',
        'per_page': top_k,
    }

    if query_vector:
        vector_str = f"embedding:([{','.join(f'{x:.5f}' for x in query_vector)}], k:{max(50, top_k * 5)}, alpha:0.7)"
        search_query['vector_query'] = vector_str

    filter_clauses = ['is_active:=true']
    if workspace_id is not None:
        filter_clauses.append(f'workspace_id:={workspace_id}')

    search_query['filter_by'] = ' && '.join(filter_clauses)

    response = client.multi_search.perform({'searches': [search_query]}, {})
    if response and 'results' in response and len(response['results']) > 0:
        return response['results'][0]
    return {'hits': []}


def ensure_analytics_collection(client: typesense.Client) -> None:
    """Ensure the dedicated Typesense collection for generic analytics datasets metadata exists."""
    schema = {
        'name': config.TYPESENSE_ANALYTICS_COLLECTION,
        'fields': [
            {'name': 'id', 'type': 'string'},
            {'name': 'workspace_id', 'type': 'int32', 'facet': True},
            {'name': 'source_id', 'type': 'string', 'facet': True},
            {'name': 'document_id', 'type': 'string', 'optional': True, 'facet': True},
            {'name': 'ingestion_purpose', 'type': 'string', 'facet': True},
            {'name': 'lifecycle_status', 'type': 'string', 'facet': True},
            {'name': 'metadata_version', 'type': 'int32'},
            {'name': 'original_filename', 'type': 'string'},
            {'name': 'dataset_summary', 'type': 'string'},
            {'name': 'sheet_names', 'type': 'string[]', 'optional': True},
            {'name': 'columns_summary', 'type': 'string'},
            {'name': 'searchable_keywords', 'type': 'string[]', 'optional': True},
            {'name': 'created_at', 'type': 'string', 'optional': True},
        ],
        'default_sorting_field': 'metadata_version',
    }
    try:
        client.collections[config.TYPESENSE_ANALYTICS_COLLECTION].retrieve()
        logger.info("Typesense collection '%s' already exists.", config.TYPESENSE_ANALYTICS_COLLECTION)
    except typesense.exceptions.ObjectNotFound:
        logger.info("Creating Typesense collection '%s'...", config.TYPESENSE_ANALYTICS_COLLECTION)
        client.collections.create(schema)
        logger.info("Collection '%s' created successfully.", config.TYPESENSE_ANALYTICS_COLLECTION)
    except Exception as e:
        logger.warning("Typesense analytics collection check warning: %s", e)


def upsert_analytics_metadata(client: typesense.Client, document: Dict[str, Any]) -> Dict[str, Any]:
    """Upsert analytical dataset metadata into Typesense."""
    ensure_analytics_collection(client)
    return client.collections[config.TYPESENSE_ANALYTICS_COLLECTION].documents.upsert(document)


def delete_analytics_metadata(client: typesense.Client, doc_id: str) -> Dict[str, Any]:
    """Delete an analytics metadata document from Typesense."""
    try:
        return client.collections[config.TYPESENSE_ANALYTICS_COLLECTION].documents[str(doc_id)].delete()
    except typesense.exceptions.ObjectNotFound:
        return {'id': str(doc_id), 'deleted': False}
    except Exception as e:
        logger.warning(f"Failed to delete analytics metadata {doc_id}: {e}")
        return {'id': str(doc_id), 'deleted': False, 'error': str(e)}


def delete_analytics_metadata_by_source(client: typesense.Client, workspace_id: int, source_id: str) -> bool:
    """Delete analytics metadata matching workspace_id and source_id."""
    try:
        filter_str = f"workspace_id:={workspace_id} && source_id:={source_id}"
        client.collections[config.TYPESENSE_ANALYTICS_COLLECTION].documents.delete({'filter_by': filter_str})
        return True
    except Exception as e:
        logger.warning(f"Failed to delete analytics metadata for source {source_id} in ws {workspace_id}: {e}")
        return False


def search_analytics_metadata(
    client: typesense.Client,
    query_text: str,
    workspace_id: int,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Search analytics metadata collection scoped strictly to workspace and active generic analytics."""
    try:
        ensure_analytics_collection(client)
        search_query = {
            'q': query_text if query_text.strip() else '*',
            'query_by': 'dataset_summary,columns_summary,sheet_names,searchable_keywords,original_filename',
            'query_by_weights': '4,4,3,3,2',
            'filter_by': f'workspace_id:={workspace_id} && lifecycle_status:=active && ingestion_purpose:=[data_analytics, generic_analytics]',
            'per_page': top_k,
        }
        res = client.collections[config.TYPESENSE_ANALYTICS_COLLECTION].documents.search(search_query)
        return res.get('hits', [])
    except Exception as e:
        logger.warning(f"Typesense search_analytics_metadata error: {e}")
        return []

