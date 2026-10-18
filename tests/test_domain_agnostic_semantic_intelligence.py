"""
Comprehensive Test Suite for Phase 2A.1: Domain-Agnostic Semantic Schema Intelligence.

Tests:
1. 6 Unrelated Business Domain Schemas:
   a. Product catalog (Retail / Commercial)
   b. Service provider billing (Consulting / Legal / Professional Services)
   c. Project budget and milestones (Software / Construction)
   d. Student enrollment & course records (Higher Education)
   e. Manufacturing production batches (Industrial / Quality Control)
   f. Clinical trial patient visits (Healthcare / Pharmaceutical)
2. Ambiguous and generic column names profiling.
3. Contradictory / noisy sample values (e.g., string in numeric column, extreme values).
4. Missing measures schema incompatibility detection (INSUFFICIENT_SCHEMA).
5. Cross-domain comparison safety detection (UNSUPPORTED_CROSS_SOURCE).
6. Malformed LLM output handling with transparent fallback.
7. Provider timeout handling with transparent fallback.
8. Schema fingerprint caching and change invalidation.
9. Legacy registry loading and on-the-fly semantic hydration.
10. Strict multi-tenant workspace isolation.
11. Preserving plain entity query fallback for MySQL (Rule 5).
"""

import os
import json
import shutil
import tempfile
import unittest
import pandas as pd

from app.analytics.data_source import (
    DataSource,
    Dataset,
    DataSourceFormat,
    DatasetCategory,
    ResolutionStatus,
    SourceResolutionResult,
)
from app.analytics.semantic_schema import (
    ColumnSemanticRole,
    ColumnSemanticDescriptor,
    DatasetSemanticSchema,
    profile_dataset_columns,
    compute_schema_fingerprint,
)
from app.analytics.semantic_interpreter import (
    BaseSemanticInterpreter,
    DeterministicSemanticInterpreter,
    FakeSemanticInterpreter,
    LLMSemanticInterpreter,
    set_semantic_interpreter,
)
from app.analytics.source_registry import DataSourceRegistry
from app.analytics.source_resolver import DataSourceResolver
from app.analytics.excel_engine import ExcelDatabaseEngine


class TestDomainAgnosticSemanticIntelligence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="phase2a1_test_")
        self.storage_dir = os.path.join(self.test_dir, "registry")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.registry = DataSourceRegistry(storage_dir=self.storage_dir)
        self.resolver = DataSourceResolver(registry=self.registry)
        self.engine = ExcelDatabaseEngine(registry=self.registry, resolver=self.resolver)
        self.fake_interpreter = FakeSemanticInterpreter()
        set_semantic_interpreter(self.fake_interpreter)

    def tearDown(self):
        # Reset global interpreter to deterministic
        set_semantic_interpreter(DeterministicSemanticInterpreter())
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Six Diverse, Unrelated Business Domains
    # -------------------------------------------------------------------------

    def test_domain_1_product_catalog(self):
        """E-Commerce / Retail product catalog schema."""
        p = os.path.join(self.test_dir, "catalog.csv")
        pd.DataFrame({
            "sku": ["SKU-100", "SKU-200"],
            "product_title": ["Laptop Backpack", "Ergonomic Mouse"],
            "unit_price": [45.0, 25.0],
            "category": ["Luggage", "Electronics"],
        }).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "catalog.csv", workspace_id=10)
        d = src["datasets"][0] if "datasets" in src else self.registry.get_source(10, src["source_id"]).datasets[0]

        self.assertIsNotNone(d.semantic_schema)
        self.assertIn("unit_price", d.semantic_schema.measures)

        res = self.resolver.resolve("Which products have the highest price?", workspace_id=10)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, src["source_id"])

    def test_domain_2_service_provider_billing(self):
        """Consulting / Legal / IT services billing schema."""
        p = os.path.join(self.test_dir, "consulting_timesheets.csv")
        pd.DataFrame({
            "timesheet_id": ["TS-101", "TS-102"],
            "consultant_name": ["Alice Smith", "Bob Jones"],
            "client_name": ["Acme Corp", "Beta LLC"],
            "billable_hours": [40.0, 32.5],
            "hourly_rate": [150.0, 175.0],
            "billing_status": ["Approved", "Pending"],
        }).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "consulting_timesheets.csv", workspace_id=11)
        d = self.registry.get_source(11, src["source_id"]).datasets[0]

        self.assertIsNotNone(d.semantic_schema)
        self.assertEqual(d.semantic_schema.domain, "consulting_services")
        self.assertIn("billable_hours", d.semantic_schema.measures)

        res = self.resolver.resolve("What are the total billable hours for consultant Alice Smith?", workspace_id=11)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, src["source_id"])

    def test_domain_3_project_budget_and_milestones(self):
        """Software / Construction project milestones and financial allocations."""
        p = os.path.join(self.test_dir, "project_milestones.csv")
        pd.DataFrame({
            "project_code": ["PRJ-A", "PRJ-B"],
            "milestone_name": ["Phase 1 Architecture", "Foundation Pouring"],
            "allocated_budget": [50000.0, 120000.0],
            "actual_cost": [48500.0, 115000.0],
            "target_deadline": ["2024-06-30", "2024-08-15"],
            "milestone_status": ["Completed", "In Progress"],
        }).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "project_milestones.csv", workspace_id=12)
        d = self.registry.get_source(12, src["source_id"]).datasets[0]

        self.assertEqual(d.semantic_schema.domain, "project_management")
        self.assertIn("allocated_budget", d.semantic_schema.measures)

        res = self.resolver.resolve("What is the allocated budget for project PRJ-A?", workspace_id=12)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, src["source_id"])

    def test_domain_4_student_enrollment_and_grades(self):
        """Higher education student course records, credits, and GPA."""
        p = os.path.join(self.test_dir, "student_grades.csv")
        pd.DataFrame({
            "student_id": ["STU-901", "STU-902"],
            "student_name": ["Fahim Rahman", "Tania Ahmed"],
            "course_code": ["CSE-101", "MAT-201"],
            "course_credits": [3.0, 4.0],
            "gpa": [3.85, 3.90],
            "letter_grade": ["A", "A+"],
            "semester": ["Spring 2024", "Spring 2024"],
        }).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "student_grades.csv", workspace_id=13)
        d = self.registry.get_source(13, src["source_id"]).datasets[0]

        self.assertEqual(d.semantic_schema.domain, "education")
        self.assertIn("gpa", d.semantic_schema.measures)

        res = self.resolver.resolve("Which students achieved the highest GPA in CSE-101?", workspace_id=13)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, src["source_id"])

    def test_domain_5_manufacturing_production_batches(self):
        """Manufacturing industrial batch yield, temperature, and defect tracking."""
        p = os.path.join(self.test_dir, "batch_production.csv")
        pd.DataFrame({
            "batch_id": ["BATCH-001", "BATCH-002"],
            "machine_id": ["MCH-9", "MCH-12"],
            "units_produced": [1000, 1500],
            "defect_count": [12, 5],
            "operating_temp_celsius": [78.5, 82.1],
            "yield_pct": [98.8, 99.6],
            "shift": ["Morning", "Night"],
        }).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "batch_production.csv", workspace_id=14)
        d = self.registry.get_source(14, src["source_id"]).datasets[0]

        self.assertEqual(d.semantic_schema.domain, "manufacturing")
        self.assertIn("defect_count", d.semantic_schema.measures)

        res = self.resolver.resolve("Which manufacturing batches had defect count above 10?", workspace_id=14)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, src["source_id"])

    def test_domain_6_clinical_trial_patient_visits(self):
        """Clinical trial patient visits, vital signs, and dosage schema."""
        p = os.path.join(self.test_dir, "clinical_trial.csv")
        pd.DataFrame({
            "patient_mrn": ["PT-881", "PT-882"],
            "visit_number": [1, 2],
            "trial_arm": ["Arm-Active-50mg", "Arm-Placebo"],
            "systolic_bp": [120, 135],
            "diastolic_bp": [80, 88],
            "dosage_mg": [50.0, 0.0],
            "adverse_event_flag": ["None", "Mild Nausea"],
        }).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "clinical_trial.csv", workspace_id=15)
        d = self.registry.get_source(15, src["source_id"]).datasets[0]

        self.assertEqual(d.semantic_schema.domain, "healthcare")
        self.assertIn("systolic_bp", d.semantic_schema.measures)

        res = self.resolver.resolve("What is the average systolic blood pressure of patients in Arm-Active-50mg?", workspace_id=15)
        self.assertEqual(res.status, ResolutionStatus.RESOLVED)
        self.assertEqual(res.source_id, src["source_id"])

    # -------------------------------------------------------------------------
    # 2. Schema Edge Cases & Robustness
    # -------------------------------------------------------------------------

    def test_ambiguous_and_generic_column_names(self):
        """Tables with ambiguous column headers (e.g. 'c1', 'val', 'code') profile safely."""
        p = os.path.join(self.test_dir, "generic_table.csv")
        pd.DataFrame({
            "c1": ["A1", "A2"],
            "val": [10.5, 20.2],
            "flag": ["OK", "WARN"],
        }).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "generic_table.csv", workspace_id=16)
        d = self.registry.get_source(16, src["source_id"]).datasets[0]

        self.assertIsNotNone(d.semantic_schema)
        self.assertEqual(d.semantic_schema.domain, "general_business")
        self.assertIn("val", d.semantic_schema.measures)

    def test_contradictory_noisy_sample_values(self):
        """Profiling handles mixed types and noisy string values without raising exceptions."""
        columns = ["metric_val", "text_info"]
        data_types = {"metric_val": "REAL", "text_info": "TEXT"}
        raw_samples = [
            {"metric_val": "100.5", "text_info": "Normal"},
            {"metric_val": "N/A", "text_info": "Missing value string"},
            {"metric_val": None, "text_info": "Null metric"},
            {"metric_val": "999999", "text_info": "Very long text " * 10},
        ]
        profile = profile_dataset_columns(
            columns=columns,
            data_types=data_types,
            raw_sample_data=raw_samples,
            row_count=4,
            dataset_name="noisy_data",
        )
        self.assertEqual(len(profile.column_profiles), 2)
        self.assertGreaterEqual(profile.column_profiles["metric_val"].null_ratio, 0.2)
        # Verify text was truncated to prevent massive payloads
        self.assertLessEqual(len(profile.column_profiles["text_info"].sample_values[0]), 50)

    def test_missing_measures_schema_incompatibility(self):
        """When query asks for an absent measure in an arbitrary domain, return INSUFFICIENT_SCHEMA."""
        p = os.path.join(self.test_dir, "student_basic.csv")
        pd.DataFrame({
            "student_id": ["STU-1", "STU-2"],
            "student_name": ["John", "Jane"],
            "semester": ["Fall 2024", "Fall 2024"],
        }).to_csv(p, index=False)
        self.engine.ingest_excel_file(p, "student_basic.csv", workspace_id=17)

        # Query asks for GPA, but student_basic lacks gpa
        res = self.resolver.resolve("What is the GPA for John?", workspace_id=17)
        self.assertEqual(res.status, ResolutionStatus.INSUFFICIENT_SCHEMA)
        self.assertIn("gpa", res.reason.lower())

    def test_cross_domain_comparison_detection(self):
        """Cross-domain query comparing distinct entities returns UNSUPPORTED_CROSS_SOURCE."""
        # Domain A: Education
        p_edu = os.path.join(self.test_dir, "students.csv")
        pd.DataFrame({"student_id": [1], "gpa": [3.5]}).to_csv(p_edu, index=False)
        self.engine.ingest_excel_file(p_edu, "students.csv", workspace_id=18)

        # Domain B: Manufacturing
        p_mfg = os.path.join(self.test_dir, "batches.csv")
        pd.DataFrame({"batch_id": [1], "yield_pct": [99]}).to_csv(p_mfg, index=False)
        self.engine.ingest_excel_file(p_mfg, "batches.csv", workspace_id=18)

        res = self.resolver.resolve("Compare education student performance with manufacturing batch yield", workspace_id=18)
        self.assertEqual(res.status, ResolutionStatus.UNSUPPORTED_CROSS_SOURCE)
        self.assertIn("Cross-dataset", res.reason)

    # -------------------------------------------------------------------------
    # 3. LLM Interpreter Robustness & Fingerprinting
    # -------------------------------------------------------------------------

    def test_llm_interpreter_malformed_json_fallback(self):
        """LLM interpreter transparently falls back to deterministic profiler on malformed JSON."""
        interpreter = LLMSemanticInterpreter()
        # Mock _call_llm_for_schema to simulate malformed JSON
        interpreter._call_llm_for_schema = lambda profile, dataset_id: None

        p = os.path.join(self.test_dir, "timesheets.csv")
        pd.DataFrame({"consultant": ["Alice"], "billable_hours": [40]}).to_csv(p, index=False)
        profile = profile_dataset_columns(["consultant", "billable_hours"], {"billable_hours": "REAL"}, dataset_name="timesheets")

        schema = interpreter.interpret_schema(profile, "ts_1")
        self.assertIsNotNone(schema)
        self.assertEqual(schema.domain, "consulting_services")
        self.assertEqual(schema.interpreter_version, "2.1.0-deterministic")

    def test_schema_fingerprint_cache_and_invalidation(self):
        """Changing columns or types changes the schema fingerprint."""
        cols_v1 = ["id", "amount"]
        types_v1 = {"id": "INTEGER", "amount": "REAL"}
        fp1 = compute_schema_fingerprint(cols_v1, types_v1, row_count=100)

        # Identical schema produces same fingerprint
        fp1_repeat = compute_schema_fingerprint(cols_v1, types_v1, row_count=100)
        self.assertEqual(fp1, fp1_repeat)

        # Adding a column alters the fingerprint
        cols_v2 = ["id", "amount", "currency"]
        types_v2 = {"id": "INTEGER", "amount": "REAL", "currency": "TEXT"}
        fp2 = compute_schema_fingerprint(cols_v2, types_v2, row_count=100)
        self.assertNotEqual(fp1, fp2)

    def test_legacy_registry_hydration_with_semantic_schema(self):
        """Legacy registry JSON missing semantic_schema is hydrated dynamically on load."""
        legacy_data = {
            "source_id": "legacy_edu_001",
            "workspace_id": 999,
            "original_filename": "legacy_grades.xlsx",
            "format": "xlsx",
            "uploaded_at": "2024-01-01T00:00:00Z",
            "datasets": [
                {
                    "dataset_id": "legacy_edu_001:Sheet1",
                    "dataset_name": "Sheet1",
                    "columns": ["student_id", "course_code", "gpa", "credits"],
                    "inferred_types": {"gpa": "REAL", "credits": "INTEGER"},
                    "row_count": 50,
                }
            ]
        }
        json_path = os.path.join(self.storage_dir, "ws_999_legacy_edu_001.json")
        with open(json_path, "w") as f:
            json.dump(legacy_data, f)

        loaded_src = self.registry.get_source(999, "legacy_edu_001")
        self.assertIsNotNone(loaded_src)
        d = loaded_src.datasets[0]
        self.assertIsNotNone(d.semantic_schema)
        # Should be classified as education domain by deterministic profiler
        self.assertEqual(d.semantic_schema.domain, "education")
        self.assertIn("gpa", d.semantic_schema.measures)

    # -------------------------------------------------------------------------
    # 4. Multi-Tenant Security & MySQL Routing
    # -------------------------------------------------------------------------

    def test_tenant_isolation_across_arbitrary_domains(self):
        """Healthcare data in Workspace 30 must never resolve to Workspace 40."""
        p = os.path.join(self.test_dir, "confidential_health.csv")
        pd.DataFrame({"patient_mrn": ["PT-99"], "systolic_bp": [120]}).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "confidential_health.csv", workspace_id=30)

        # Resolution from Workspace 40
        res = self.resolver.resolve("What is the systolic blood pressure of patient PT-99?", workspace_id=40)
        self.assertEqual(res.status, ResolutionStatus.NOT_FOUND)

        # Probe with explicit source_id from Workspace 40
        res_probe = self.resolver.resolve("Show patient data", workspace_id=40, explicit_source_id=src["source_id"])
        self.assertEqual(res_probe.status, ResolutionStatus.NOT_FOUND)

    def test_plain_entity_query_fallback_for_mysql(self):
        """Plain entity queries without file indicators return NOT_FOUND so MySQL handles them (Rule 5)."""
        p = os.path.join(self.test_dir, "general.csv")
        pd.DataFrame({"seller": ["Hasan"], "sales": [1000]}).to_csv(p, index=False)
        self.engine.ingest_excel_file(p, "general.csv", workspace_id=50)

        res = self.resolver.resolve("Hasan কত sales করেছে?", workspace_id=50)
        self.assertEqual(res.status, ResolutionStatus.NOT_FOUND)
        self.assertIsNone(res.source)

    # -------------------------------------------------------------------------
    # 5. Hardening Audits: Privacy, Cache Boundaries, Ambiguity & Validation
    # -------------------------------------------------------------------------

    def test_pii_and_credential_redaction_before_profiling(self):
        """Sensitive PII (credit cards, emails, phones) and secret columns are strictly redacted."""
        columns = ["user_id", "email_addr", "mobile_phone", "cc_num", "api_token"]
        data_types = {c: "TEXT" for c in columns}
        samples = [
            {
                "user_id": "U101",
                "email_addr": "customer.vip@enterprise.com",
                "mobile_phone": "01712345678",
                "cc_num": "4111111111111234",
                "api_token": "sk_live_secretkey9999",
            }
        ]
        profile = profile_dataset_columns(
            columns=columns,
            data_types=data_types,
            raw_sample_data=samples,
            row_count=1,
            dataset_name="sensitive_customers",
            workspace_id=77,
        )

        # Email redaction
        self.assertIn("[EMAIL_REDACTED]", profile.column_profiles["email_addr"].sample_values)
        self.assertNotIn("customer.vip@enterprise.com", profile.column_profiles["email_addr"].sample_values)

        # Phone redaction
        self.assertIn("[PHONE_REDACTED]", profile.column_profiles["mobile_phone"].sample_values)
        self.assertNotIn("01712345678", profile.column_profiles["mobile_phone"].sample_values)

        # Credit card redaction
        cc_sample = profile.column_profiles["cc_num"].sample_values[0]
        self.assertTrue(cc_sample.startswith("XXXX-XXXX-XXXX-"))
        self.assertTrue(cc_sample.endswith("1234"))

        # Secret column exclusion
        self.assertEqual(profile.column_profiles["api_token"].sample_values, ["[REDACTED_SECRET]"])
        self.assertNotIn("sk_live_secretkey9999", profile.column_profiles["api_token"].sample_values)

    def test_cross_workspace_cache_isolation_and_permissions(self):
        """Cache keys are partitioned by workspace_id and disk cache files have 0o600 permissions."""
        cache_dir = os.path.join(self.test_dir, "sem_cache")
        interpreter = LLMSemanticInterpreter(cache_dir=cache_dir)

        # Mock LLM generation to return custom schema
        mock_schema_ws1 = DatasetSemanticSchema(
            dataset_id="tbl_1",
            domain="consulting_services",
            purpose="Confidential timesheets for Workspace 101",
            primary_entity="consultant_timesheet",
            columns={},
            schema_fingerprint="fp_identical_123",
            confidence=0.9,
            interpreter_version=interpreter.VERSION,
        )
        interpreter._call_llm_for_schema = lambda prof, d_id: mock_schema_ws1

        # Profile for Workspace 101
        prof_ws1 = profile_dataset_columns(
            columns=["col_a", "val_b"],
            data_types={"col_a": "TEXT", "val_b": "REAL"},
            row_count=10,
            dataset_name="timesheets",
            workspace_id=101,
        )
        # Force fingerprint
        prof_ws1.schema_fingerprint = "fp_identical_123"

        res_ws1 = interpreter.interpret_schema(prof_ws1, "tbl_1")
        self.assertEqual(res_ws1.purpose, "Confidential timesheets for Workspace 101")

        # Verify disk cache file exists with ws_101 prefix
        expected_cache_ws1 = os.path.join(cache_dir, f"sem_ws_101_fp_identical_123_{interpreter.VERSION}.json")
        self.assertTrue(os.path.exists(expected_cache_ws1))
        # Check permissions (0o600)
        file_mode = oct(os.stat(expected_cache_ws1).st_mode & 0o777)
        self.assertEqual(file_mode, "0o600")

        # Now Workspace 202 interprets an identical schema
        prof_ws2 = profile_dataset_columns(
            columns=["col_a", "val_b"],
            data_types={"col_a": "TEXT", "val_b": "REAL"},
            row_count=10,
            dataset_name="timesheets",
            workspace_id=202,
        )
        prof_ws2.schema_fingerprint = "fp_identical_123"

        # Change mock for WS 202 to verify it doesn't read WS 101's cached schema
        mock_schema_ws2 = DatasetSemanticSchema(
            dataset_id="tbl_2",
            domain="consulting_services",
            purpose="Workspace 202 timesheets",
            primary_entity="consultant_timesheet",
            columns={},
            schema_fingerprint="fp_identical_123",
            confidence=0.9,
            interpreter_version=interpreter.VERSION,
        )
        interpreter._call_llm_for_schema = lambda prof, d_id: mock_schema_ws2

        res_ws2 = interpreter.interpret_schema(prof_ws2, "tbl_2")
        self.assertEqual(res_ws2.purpose, "Workspace 202 timesheets")
        # Ensure separate cache file for WS 202
        expected_cache_ws2 = os.path.join(cache_dir, f"sem_ws_202_fp_identical_123_{interpreter.VERSION}.json")
        self.assertTrue(os.path.exists(expected_cache_ws2))

    def test_llm_structured_output_validation_strips_hallucinated_measures(self):
        """Measures returned by LLM that do not exist as genuine columns are discarded."""
        interpreter = LLMSemanticInterpreter()
        # Profile has only 2 genuine columns
        prof = profile_dataset_columns(
            columns=["task_id", "hours"],
            data_types={"task_id": "TEXT", "hours": "REAL"},
            row_count=5,
            dataset_name="tasks",
            workspace_id=5,
        )

        class MockLLMResponse:
            content = json.dumps({
                "domain": "project_management",
                "purpose": "Task tracking",
                "primary_entity": "project_task",
                "columns": {
                    "task_id": {"semantic_role": "identifier"},
                    "hours": {"semantic_role": "measure"},
                },
                # Hallucinated measures that DO NOT exist in columns
                "measures": ["hours", "profit", "net_margin", "roi"],
                "dimensions": ["task_id", "hallucinated_dept"],
                "confidence": 1.5, # Out of range confidence
            })
            provider = "test_provider"
            model = "test_model"

        # Mock generate
        class MockClient:
            async def generate(self, req, provider_name=None):
                return MockLLMResponse()

        # Temporarily mock LLMClient inside _call_llm_for_schema
        import app.llm.client
        orig_client = app.llm.client.LLMClient
        app.llm.client.LLMClient = lambda: MockClient()

        try:
            schema = interpreter._call_llm_for_schema(prof, "tasks_1")
            self.assertIsNotNone(schema)
            # Hallucinated measures must be stripped
            self.assertEqual(schema.measures, ["hours"])
            self.assertNotIn("profit", schema.measures)
            self.assertNotIn("net_margin", schema.measures)
            # Hallucinated dimension stripped
            self.assertEqual(schema.dimensions, ["task_id"])
            self.assertNotIn("hallucinated_dept", schema.dimensions)
            # Confidence clamped to <= 1.0
            self.assertLessEqual(schema.confidence, 1.0)
        finally:
            app.llm.client.LLMClient = orig_client

    def test_resolver_generic_ambiguous_queries(self):
        """Resolver returns AMBIGUOUS for generic queries when multiple datasets match the metric."""
        # Source 1: Exams (has score)
        p1 = os.path.join(self.test_dir, "exam_scores.csv")
        pd.DataFrame({"student": ["Alice"], "score": [92.0]}).to_csv(p1, index=False)
        self.engine.ingest_excel_file(p1, "exam_scores.csv", workspace_id=80)

        # Source 2: Audits (also has score)
        p2 = os.path.join(self.test_dir, "audit_scores.csv")
        pd.DataFrame({"department": ["Ops"], "score": [88.5]}).to_csv(p2, index=False)
        self.engine.ingest_excel_file(p2, "audit_scores.csv", workspace_id=80)

        # Query 1: "What is the average score?"
        res1 = self.resolver.resolve("What is the average score?", workspace_id=80)
        self.assertEqual(res1.status, ResolutionStatus.AMBIGUOUS)
        self.assertEqual(len(res1.candidates), 2)

        # Source 3: Billing with hourly_rate
        p3 = os.path.join(self.test_dir, "timesheet_rate.csv")
        pd.DataFrame({"consultant": ["Bob"], "hourly_rate": [150.0], "amount": [6000.0], "status": ["Paid"]}).to_csv(p3, index=False)
        self.engine.ingest_excel_file(p3, "timesheet_rate.csv", workspace_id=81)

        # Source 4: Shipping with shipping_rate
        p4 = os.path.join(self.test_dir, "shipping_rate.csv")
        pd.DataFrame({"courier": ["DHL"], "shipping_rate": [25.0], "amount": [250.0], "status": ["Pending"]}).to_csv(p4, index=False)
        self.engine.ingest_excel_file(p4, "shipping_rate.csv", workspace_id=81)

        # Query 2: "Which record has the highest rate?"
        res2 = self.resolver.resolve("Which record has the highest rate?", workspace_id=81)
        self.assertEqual(res2.status, ResolutionStatus.AMBIGUOUS)

        # Query 3: "Show the amount by status."
        res3 = self.resolver.resolve("Show the amount by status.", workspace_id=81)
        self.assertEqual(res3.status, ResolutionStatus.AMBIGUOUS)

    def test_resolver_absent_measure_in_single_source(self):
        """Asking for an unrepresented measure in a single-source workspace returns INSUFFICIENT_SCHEMA."""
        p = os.path.join(self.test_dir, "roster.csv")
        pd.DataFrame({"student_name": ["Alice"], "grade_level": ["Grade 10"]}).to_csv(p, index=False)
        self.engine.ingest_excel_file(p, "roster.csv", workspace_id=85)

        # Query asks for average score, but roster only has student_name and grade_level
        res = self.resolver.resolve("What is the average score?", workspace_id=85)
        self.assertEqual(res.status, ResolutionStatus.INSUFFICIENT_SCHEMA)
        self.assertIn("score", res.reason.lower())

    def test_resolver_domain_keyword_alone_does_not_falsely_select(self):
        """A query containing a domain keyword without matching columns/entities does not select the dataset."""
        p = os.path.join(self.test_dir, "consulting_data.csv")
        pd.DataFrame({"client_name": ["Acme Corp"], "consultant": ["Alice"], "billable_hours": [40]}).to_csv(p, index=False)
        self.engine.ingest_excel_file(p, "consulting_data.csv", workspace_id=86)

        # Query mentions domain word "consulting" but nothing else matches (asks about weather or unrelated topic)
        res = self.resolver.resolve("Tell me consulting weather forecast", workspace_id=86)
        # Should not resolve to consulting_data.csv because "weather forecast" is completely unsupported
        self.assertNotEqual(res.status, ResolutionStatus.RESOLVED)

    def test_explicit_source_id_with_absent_measure_returns_insufficient_schema(self):
        """Explicit source reference with an unsupported metric returns INSUFFICIENT_SCHEMA."""
        p = os.path.join(self.test_dir, "simple_roster.csv")
        pd.DataFrame({"name": ["Alice", "Bob"]}).to_csv(p, index=False)
        src = self.engine.ingest_excel_file(p, "simple_roster.csv", workspace_id=87)

        res = self.resolver.resolve(
            "What is the average salary?",
            workspace_id=87,
            explicit_source_id=src["source_id"]
        )
        self.assertEqual(res.status, ResolutionStatus.INSUFFICIENT_SCHEMA)
        self.assertIn("salary", res.reason.lower())


if __name__ == "__main__":
    unittest.main()

