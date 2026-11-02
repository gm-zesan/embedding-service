import unittest
from app.analytics.semantic_schema import (
    CandidateRelationshipDescriptor,
    DiscoveredRelationshipEdge,
    SpreadsheetRelationshipGraph,
    DatasetSemanticSchema,
    RelationshipCardinality,
    RelationshipLifecycleState,
)


class TestRelationshipContract230(unittest.TestCase):
    """
    Contract 2.3.0 serialization/deserialization symmetry tests between Laravel and Python.
    """

    def test_candidate_relationship_descriptor_from_laravel_payload(self):
        # Payload format produced by Laravel SpreadsheetRelationshipGraph::toPythonCandidateRelationships()
        laravel_payload = {
            "id": "edge_abcdef123456",
            "from_sheet": "Orders",
            "from_column": "customer_id",
            "from_column_ids": ["col_001"],
            "to_sheet": "Customers",
            "to_column": "id",
            "to_column_ids": ["col_000"],
            "cardinality": "1:N",
            "confidence": 0.95,
            "is_junction": False,
            "inclusion_ratio": 1.0,
            "state": "DISCOVERED",
            "contract_version": "2.3.0",
        }

        desc = CandidateRelationshipDescriptor(**laravel_payload)
        self.assertEqual(desc.id, "edge_abcdef123456")
        self.assertEqual(desc.from_sheet, "Orders")
        self.assertEqual(desc.to_sheet, "Customers")
        self.assertEqual(desc.cardinality, RelationshipCardinality.ONE_TO_MANY.value)
        self.assertEqual(desc.state, RelationshipLifecycleState.DISCOVERED.value)
        self.assertEqual(desc.contract_version, "2.3.0")

    def test_discovered_relationship_edge_from_laravel_array(self):
        # Payload format produced by Laravel DiscoveredRelationshipEdge::toArray()
        laravel_edge_array = {
            "id": "edge_1234567890abcdef",
            "source_sheet": "Clinical_Appointments",
            "source_column_ids": ["col_001"],
            "source_headers": ["attending_doc_ref"],
            "target_sheet": "Provider_Roster",
            "target_column_ids": ["col_000"],
            "target_headers": ["doc_ref_no"],
            "cardinality": "1:N",
            "is_junction_table": False,
            "inclusion_ratio": 1.0,
            "header_similarity": 0.85,
            "entity_alignment": 0.90,
            "type_compatibility": 1.0,
            "confidence": 0.965,
            "state": "CONFIRMED",
            "evidence": ["Inclusion Dependency (IND): 1.0 (VALID ≥ 0.85)"],
            "orphan_count": 0,
            "confirmed_by": 42,
            "confirmed_at": "2026-11-02T13:40:00Z",
            "rejection_reason": None,
        }

        edge = DiscoveredRelationshipEdge(**laravel_edge_array)
        self.assertEqual(edge.id, "edge_1234567890abcdef")
        self.assertEqual(edge.source_sheet, "Clinical_Appointments")
        self.assertEqual(edge.target_sheet, "Provider_Roster")
        self.assertEqual(edge.cardinality, "1:N")
        self.assertEqual(edge.state, "CONFIRMED")
        self.assertEqual(edge.confirmed_by, 42)

    def test_spreadsheet_relationship_graph_from_laravel_array(self):
        # Payload format produced by Laravel SpreadsheetRelationshipGraph::toArray()
        laravel_graph_array = {
            "graph_version": "2.3.0",
            "workspace_id": 1,
            "document_id": "doc-uuid-9999",
            "job_id": "job-uuid-1111",
            "generation_id": "gen-uuid-2222",
            "nodes": {
                "Departments": {"sheet_name": "Departments", "row_count": 5},
                "Employees": {"sheet_name": "Employees", "row_count": 50},
            },
            "edges": {
                "edge_1": {
                    "id": "edge_1",
                    "source_sheet": "Employees",
                    "source_column_ids": ["col_dept"],
                    "source_headers": ["dept_code"],
                    "target_sheet": "Departments",
                    "target_column_ids": ["col_code"],
                    "target_headers": ["code"],
                    "cardinality": "1:N",
                    "is_junction_table": False,
                    "inclusion_ratio": 1.0,
                    "header_similarity": 0.9,
                    "entity_alignment": 0.9,
                    "type_compatibility": 1.0,
                    "confidence": 0.97,
                    "state": "CONFIRMED",
                    "evidence": [],
                    "orphan_count": 0,
                }
            },
            "ambiguities": [],
            "discovered_at": "2026-11-02T14:00:00Z",
        }

        graph = SpreadsheetRelationshipGraph(**laravel_graph_array)
        self.assertEqual(graph.graph_version, "2.3.0")
        self.assertEqual(graph.workspace_id, 1)
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges["edge_1"].source_sheet, "Employees")

    def test_backward_compatibility_with_2_1_0_schema(self):
        # 2.1.0 schema without relationship graph or with legacy candidate_relationships
        legacy_payload = {
            "dataset_id": "ds_legacy_001",
            "domain": "e_commerce",
            "purpose": "Legacy store dataset",
            "schema_fingerprint": "hash_abc_123",
            "interpreter_version": "2.1.0",
            "candidate_relationships": [
                {"from": "orders.cust_id", "to": "customers.id"}
            ],
        }

        schema = DatasetSemanticSchema(**legacy_payload)
        self.assertEqual(schema.interpreter_version, "2.1.0")
        self.assertEqual(len(schema.candidate_relationships), 1)
        self.assertIsNone(schema.relationship_graph)


if __name__ == "__main__":
    unittest.main()

