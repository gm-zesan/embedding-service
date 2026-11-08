"""
Tests for VirtualRelationalCompiler under Contract 3.0.0.
Verifies deterministic SQL compilation, parameter binding, confirmed-only join resolution,
Bangla/non-ASCII mapping, and rejection of unconfirmed/stale relationships.
"""

import unittest
from app.analytics.virtual_compiler import VirtualRelationalCompiler, VirtualCompilerError
from app.analytics.contract_v3 import (
    DynamicSchemaCatalog,
    VirtualTableMeta,
    VirtualColumnMeta,
    VirtualColumnType,
)
from app.analytics.models import (
    SemanticQueryPlan,
    DomainEnum,
    MeasureSpec,
    DimensionSpec,
    FilterSpec,
    OrderBySpec,
    AggregationType,
    OperatorType,
)
from app.analytics.semantic_schema import (
    SpreadsheetRelationshipGraph,
    DiscoveredRelationshipEdge,
    RelationshipLifecycleState,
    RelationshipCardinality,
)


class TestVirtualCompiler(unittest.TestCase):

    def _build_test_catalog(self) -> DynamicSchemaCatalog:
        """Helper to build a 2-table catalog (Orders and Customers) with Bangla headers."""
        orders_table = VirtualTableMeta(
            name="tbl_orders",
            sheet_name="Orders",
            row_count=100,
            columns={
                "col_order_id": VirtualColumnMeta(name="col_order_id", original_header="অর্ডার আইডি", physical_type=VirtualColumnType.TEXT, is_primary_key=True),
                "col_cust_id": VirtualColumnMeta(name="col_cust_id", original_header="গ্রাহক আইডি", physical_type=VirtualColumnType.TEXT),
                "col_amount": VirtualColumnMeta(name="col_amount", original_header="মোট টাকা", physical_type=VirtualColumnType.NUMERIC),
                "col_status": VirtualColumnMeta(name="col_status", original_header="স্ট্যাটাস", physical_type=VirtualColumnType.TEXT),
            },
            primary_key=["col_order_id"],
        )

        customers_table = VirtualTableMeta(
            name="tbl_customers",
            sheet_name="Customers",
            row_count=50,
            columns={
                "col_customer_id": VirtualColumnMeta(name="col_customer_id", original_header="Customer ID", physical_type=VirtualColumnType.TEXT, is_primary_key=True),
                "col_cust_name": VirtualColumnMeta(name="col_cust_name", original_header="গ্রাহকের নাম", physical_type=VirtualColumnType.TEXT),
                "col_city": VirtualColumnMeta(name="col_city", original_header="শহর", physical_type=VirtualColumnType.TEXT),
            },
            primary_key=["col_customer_id"],
        )

        return DynamicSchemaCatalog(
            workspace_id=1,
            dataset_id="test_ds",
            generation_id="gen_001",
            schema_fingerprint="fp_schema_test",
            tables={
                "tbl_orders": orders_table,
                "tbl_customers": customers_table,
            },
        )

    def test_single_table_parameterized_compilation(self):
        """Verifies single-table aggregation compiles deterministically with ? placeholders."""
        catalog = self._build_test_catalog()
        compiler = VirtualRelationalCompiler()

        plan = SemanticQueryPlan(
            domain=DomainEnum.SALES,
            measures=[MeasureSpec(name="col_amount", aggregation=AggregationType.SUM, alias="total_sales")],
            filters=[FilterSpec(field="col_status", operator=OperatorType.EQUALS, value="Delivered")],
            limit=25,
        )

        artifact = compiler.compile(plan=plan, catalog=catalog)

        self.assertIn("SELECT", artifact.sql_template)
        self.assertIn("SUM(tbl_orders.col_amount) AS total_sales", artifact.sql_template)
        self.assertIn("FROM tbl_orders", artifact.sql_template)
        self.assertIn("tbl_orders.col_status = ?", artifact.sql_template)
        self.assertNotIn("Delivered", artifact.sql_template)  # Strict parameter binding!
        self.assertEqual(artifact.params, ["Delivered"])
        self.assertEqual(artifact.referenced_tables, ["tbl_orders"])
        self.assertEqual(set(artifact.referenced_columns), {"col_amount", "col_status"})
        self.assertEqual(len(artifact.query_fingerprint), 64)

    def test_confirmed_join_synthesis(self):
        """Verifies multi-sheet join synthesis occurs when relationship edge is CONFIRMED."""
        catalog = self._build_test_catalog()
        compiler = VirtualRelationalCompiler()

        confirmed_edge = DiscoveredRelationshipEdge(
            id="edge_orders_customers_001",
            source_sheet="Orders",
            source_headers=["গ্রাহক আইডি"],
            source_column_ids=["col_cust_id"],
            target_sheet="Customers",
            target_headers=["Customer ID"],
            target_column_ids=["col_customer_id"],
            cardinality=RelationshipCardinality.ONE_TO_MANY.value,
            confidence=0.95,
            state=RelationshipLifecycleState.CONFIRMED.value,
        )
        graph = SpreadsheetRelationshipGraph(
            workspace_id=1,
            document_id="test_ds",
            generation_id="gen_001",
            edges={confirmed_edge.id: confirmed_edge},
        )

        plan = SemanticQueryPlan(
            domain=DomainEnum.SALES,
            measures=[MeasureSpec(name="col_amount", aggregation=AggregationType.SUM, alias="total_sales")],
            dimensions=[DimensionSpec(name="col_cust_name", alias="customer_name")],
            group_by=["col_cust_name"],
            order_by=[OrderBySpec(field="col_amount", direction="desc")],
            limit=10,
        )

        artifact = compiler.compile(plan=plan, catalog=catalog, relationship_graph=graph)

        self.assertTrue("JOIN tbl_customers" in artifact.sql_template or "JOIN tbl_orders" in artifact.sql_template)
        self.assertTrue("tbl_orders.col_cust_id = tbl_customers.col_customer_id" in artifact.sql_template or "tbl_customers.col_customer_id = tbl_orders.col_cust_id" in artifact.sql_template)
        self.assertIn("GROUP BY tbl_customers.col_cust_name", artifact.sql_template)
        self.assertEqual(set(artifact.referenced_tables), {"tbl_orders", "tbl_customers"})
        self.assertIn(confirmed_edge.id, artifact.confirmed_edge_ids)

    def test_rejection_of_unconfirmed_edges_prevents_cartesian_join(self):
        """Verifies that an unconfirmed edge (DISCOVERED / NEEDS_CONFIRMATION) is rejected and raises an error."""
        catalog = self._build_test_catalog()
        compiler = VirtualRelationalCompiler()

        unconfirmed_edge = DiscoveredRelationshipEdge(
            id="edge_orders_customers_unconfirmed",
            source_sheet="Orders",
            source_headers=["গ্রাহক আইডি"],
            source_column_ids=["col_cust_id"],
            target_sheet="Customers",
            target_headers=["Customer ID"],
            target_column_ids=["col_customer_id"],
            cardinality=RelationshipCardinality.ONE_TO_MANY.value,
            confidence=0.60,
            state=RelationshipLifecycleState.DISCOVERED.value,  # NOT CONFIRMED!
        )
        graph = SpreadsheetRelationshipGraph(
            workspace_id=1,
            document_id="test_ds",
            generation_id="gen_001",
            edges={unconfirmed_edge.id: unconfirmed_edge},
        )

        plan = SemanticQueryPlan(
            domain=DomainEnum.SALES,
            measures=[MeasureSpec(name="col_amount", aggregation=AggregationType.SUM)],
            dimensions=[DimensionSpec(name="col_cust_name")],
        )

        with self.assertRaises(VirtualCompilerError):
            compiler.compile(plan=plan, catalog=catalog, relationship_graph=graph)


if __name__ == "__main__":
    unittest.main()
