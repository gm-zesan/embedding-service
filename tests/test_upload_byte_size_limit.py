import io
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
import app.config as config
import app.main as main_mod


class TestUploadByteSizeLimit(unittest.TestCase):
    """
    Focused verification tests for streaming raw upload byte-size limits in
    app/analytics/router.py and app/main.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_key = "test-auth-secret-key-12345"

    def setUp(self):
        # Configure test key for each test
        self.orig_config_key = config.API_KEY
        self.orig_config_env = config.APP_ENV
        self.orig_main_key = main_mod.API_KEY
        self.orig_main_env = main_mod.APP_ENV
        self.orig_max_upload = getattr(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", 20 * 1024 * 1024)

        config.API_KEY = self.test_key
        config.APP_ENV = "testing"
        main_mod.API_KEY = self.test_key
        main_mod.APP_ENV = "testing"

        self.auth_headers = {"X-API-Key": self.test_key}

    def tearDown(self):
        config.API_KEY = self.orig_config_key
        config.APP_ENV = self.orig_config_env
        main_mod.API_KEY = self.orig_main_key
        main_mod.APP_ENV = self.orig_main_env
        config.ANALYTICS_MAX_UPLOAD_SIZE_BYTES = self.orig_max_upload

    def test_default_limit_is_20_mib(self):
        """Default raw upload limit is 20 MiB (20,971,520 bytes)."""
        expected_bytes = 20 * 1024 * 1024
        self.assertEqual(config.ANALYTICS_MAX_UPLOAD_SIZE_BYTES, expected_bytes)

    def test_under_limit_upload_succeeds(self):
        """Upload strictly under the byte limit is accepted and processed."""
        csv_content = b"item,price\nWidgetA,100\nWidgetB,200\n"
        csv_size = len(csv_content)

        # Set limit comfortably above content size
        with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", csv_size + 100):
            response = self.client.post(
                "/analytics/sources/upload",
                files={"file": ("test_under.csv", io.BytesIO(csv_content), "text/csv")},
                data={"workspace_id": 1},
                headers=self.auth_headers,
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("filename"), "test_under.csv")
        self.assertEqual(data.get("total_rows"), 2)

    def test_exact_limit_upload_succeeds(self):
        """Upload having byte size exactly equal to the configured limit succeeds."""
        csv_content = b"col1,col2\nval1,val2\nval3,val4\n"
        exact_size = len(csv_content)

        # Set limit to the exact byte count
        with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", exact_size):
            response = self.client.post(
                "/analytics/sources/upload",
                files={"file": ("test_exact.csv", io.BytesIO(csv_content), "text/csv")},
                data={"workspace_id": 1},
                headers=self.auth_headers,
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))

    def test_oversized_upload_rejected_with_413(self):
        """Upload exceeding the byte limit by even 1 byte is rejected with HTTP 413."""
        csv_content = b"col1,col2\nval1,val2\nval3,val4\n"
        limit = len(csv_content) - 1  # 1 byte smaller than file

        with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", limit):
            response = self.client.post(
                "/analytics/sources/upload",
                files={"file": ("test_over.csv", io.BytesIO(csv_content), "text/csv")},
                data={"workspace_id": 1},
                headers=self.auth_headers,
            )

        self.assertEqual(response.status_code, 413)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("exceeds maximum allowed size", data["detail"])
        self.assertIn(str(limit), data["detail"])

    def test_temporary_file_cleaned_up_on_413_rejection(self):
        """Temporary file created during streaming is deleted when upload exceeds limit."""
        temp_dir = tempfile.gettempdir()
        initial_temp_files = set(os.listdir(temp_dir))

        # 1 MiB payload, limit 1 KiB
        payload = b"header1,header2\n" + (b"data_value_x,data_value_y\n" * 40000)
        limit = 1024  # 1 KiB limit

        with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", limit):
            response = self.client.post(
                "/analytics/sources/upload",
                files={"file": ("large_file.csv", io.BytesIO(payload), "text/csv")},
                data={"workspace_id": 1},
                headers=self.auth_headers,
            )

        self.assertEqual(response.status_code, 413)

        # Confirm no orphaned .csv temporary files were left in temp_dir
        final_temp_files = set(os.listdir(temp_dir))
        new_files = final_temp_files - initial_temp_files
        csv_orphans = [f for f in new_files if f.endswith(".csv")]
        self.assertEqual(
            csv_orphans,
            [],
            f"Expected zero orphaned temporary files after HTTP 413, found: {csv_orphans}",
        )

    def test_streaming_early_abort_in_router(self):
        """Router aborts reading immediately once byte threshold is exceeded."""
        import asyncio
        from fastapi import UploadFile, HTTPException
        from app.analytics.router import upload_analytical_source

        chunks_read = []

        class MockFile:
            def __init__(self, total_chunks=50, chunk_size=64 * 1024):
                self.chunks_left = total_chunks
                self.chunk_size = chunk_size

            def read(self, size=-1):
                if self.chunks_left <= 0:
                    return b""
                self.chunks_left -= 1
                chunks_read.append(1)
                return b"a" * min(size if size > 0 else self.chunk_size, self.chunk_size)

            def close(self):
                pass

        mock_file = MockFile(total_chunks=50, chunk_size=64 * 1024)
        upload = UploadFile(filename="test_stream.csv", file=mock_file)

        # Set limit to 2 chunks (128 KiB)
        limit = 128 * 1024
        with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", limit):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(upload_analytical_source(file=upload, workspace_id=1))

        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("exceeds maximum allowed size", ctx.exception.detail)
        # Bounded read reads chunk 1 (64K), chunk 2 (64K), and then exactly 1 byte on chunk 3!
        self.assertEqual(
            len(chunks_read),
            3,
            f"Expected early abort at exactly 3 chunks, but read {len(chunks_read)} chunks",
        )

    def test_maximum_extra_bytes_read_is_exactly_one_byte(self):
        """Quantify: the maximum extra bytes read past the limit is strictly 1 byte."""
        import asyncio
        from fastapi import UploadFile, HTTPException
        from app.analytics.router import upload_analytical_source

        limit = 500  # 500 bytes limit
        total_bytes_requested = []

        class ExactMeteredFile:
            def __init__(self, total_bytes=2000):
                self.data = b"x" * total_bytes
                self.pos = 0

            def read(self, size=-1):
                if self.pos >= len(self.data):
                    return b""
                read_size = min(size, len(self.data) - self.pos)
                total_bytes_requested.append(read_size)
                chunk = self.data[self.pos : self.pos + read_size]
                self.pos += read_size
                return chunk

            def close(self):
                pass

        metered = ExactMeteredFile(total_bytes=2000)
        upload = UploadFile(filename="test_meter.csv", file=metered)

        with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", limit):
            with self.assertRaises(HTTPException):
                asyncio.run(upload_analytical_source(file=upload, workspace_id=1))

        # Sum of bytes read by the router
        total_read = sum(total_bytes_requested)
        extra_read = total_read - limit

        self.assertEqual(
            extra_read,
            1,
            f"Expected strictly 1 extra byte read past configured limit, but read {extra_read} extra bytes",
        )
        self.assertEqual(total_read, limit + 1)

    def test_invalid_zero_and_negative_config_fallbacks_to_positive_default(self):
        """Invalid, zero, or negative config values fail-safe to validated 20 MiB default."""
        csv_content = b"item,price\nItem1,50\n"

        # Test zero, negative, and invalid string
        invalid_values = [0, -100, "not-a-number", None]

        for val in invalid_values:
            with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", val):
                response = self.client.post(
                    "/analytics/sources/upload",
                    files={"file": ("test_fallback.csv", io.BytesIO(csv_content), "text/csv")},
                    data={"workspace_id": 1},
                    headers=self.auth_headers,
                )
                self.assertEqual(
                    response.status_code,
                    200,
                    f"Config value {val!r} must fail-safe to positive default, but got status {response.status_code}",
                )

    def test_unsupported_format_rejected_before_streaming(self):
        """Unsupported extensions return 400 Bad Request."""
        response = self.client.post(
            "/analytics/sources/upload",
            files={"file": ("malicious.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
            data={"workspace_id": 1},
            headers=self.auth_headers,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file format", response.json()["detail"])

    def test_middleware_rejects_oversized_content_length_before_multipart_parsing(self):
        """ASGI middleware returns HTTP 413 immediately when Content-Length exceeds limit."""
        # 30 MiB Content-Length header (exceeds 20 MiB + 1 MiB allowance)
        headers = dict(self.auth_headers)
        headers["Content-Length"] = str(30 * 1024 * 1024)

        response = self.client.post(
            "/analytics/sources/upload",
            headers=headers,
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("Request too large", response.json().get("detail", ""))

    def test_middleware_allows_analytics_upload_larger_than_general_10mb(self):
        """Middleware allows 15 MiB analytics upload without being blocked by general 10MB limit."""
        # 15 MiB Content-Length header is under 20 MiB + 1 MiB allowance
        headers = dict(self.auth_headers)
        headers["Content-Length"] = str(15 * 1024 * 1024)

        # Send empty/dummy post to check middleware doesn't reject with 413
        # Will fail validation on missing fields (422) instead of 413!
        response = self.client.post(
            "/analytics/sources/upload",
            headers=headers,
        )
        # If it reached 422 (unprocessable entity on form data), middleware passed it!
        self.assertNotEqual(response.status_code, 413)
        self.assertEqual(response.status_code, 422)

    def test_middleware_rejects_malformed_content_length_with_400(self):
        """Malformed Content-Length headers return HTTP 400 Bad Request."""
        response = self.client.post(
            "/analytics/sources/upload",
            headers={"Content-Length": "invalid_length"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "Invalid Content-Length header")

    def test_middleware_rejects_negative_content_length_with_400(self):
        """Negative Content-Length headers return HTTP 400 Bad Request."""
        response = self.client.post(
            "/analytics/sources/upload",
            headers={"Content-Length": "-500"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("detail"), "Invalid negative Content-Length header")

    def test_middleware_rejects_duplicate_conflicting_content_length_with_400(self):
        """Conflicting duplicate Content-Length headers return HTTP 400 Bad Request."""
        import asyncio
        import json

        async def run_duplicate_cl():
            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            status_code = None
            body_chunks = []
            async def send(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                elif message["type"] == "http.response.body":
                    body_chunks.append(message.get("body", b""))

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/analytics/sources/upload",
                "query_string": b"",
                "headers": [
                    (b"host", b"testserver"),
                    (b"content-length", b"100"),
                    (b"content-length", b"200"),
                ],
            }
            await app(scope, receive, send)
            body = b"".join(body_chunks).decode("utf-8", errors="replace")
            return status_code, json.loads(body)

        status, data = asyncio.run(run_duplicate_cl())
        self.assertEqual(status, 400)
        self.assertEqual(data.get("detail"), "Invalid duplicate Content-Length header")

    def test_chunked_streaming_under_limit_accepted(self):
        """Chunked multipart stream without Content-Length below limit is parsed and accepted."""
        import asyncio
        import json

        boundary = "----TestBoundaryUnderLimit"
        part_header = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"workspace_id\"\r\n\r\n"
            f"1\r\n"
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"test_chunked.csv\"\r\n"
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8")
        chunk_data = b"col1,col2\nval1,val2\n"
        chunk_footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        chunks = [part_header, chunk_data, chunk_footer]

        async def run_chunked_under():
            chunks_read = 0
            idx = 0

            async def receive():
                nonlocal chunks_read, idx
                if idx >= len(chunks):
                    return {"type": "http.request", "body": b"", "more_body": False}
                chunk = chunks[idx]
                idx += 1
                chunks_read += 1
                return {"type": "http.request", "body": chunk, "more_body": (idx < len(chunks))}

            status_code = None
            body_chunks = []
            async def send(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                elif message["type"] == "http.response.body":
                    body_chunks.append(message.get("body", b""))

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/analytics/sources/upload",
                "query_string": b"",
                "headers": [
                    (b"host", b"testserver"),
                    (b"x-api-key", self.test_key.encode("utf-8")),
                    (b"content-type", f"multipart/form-data; boundary={boundary}".encode("utf-8")),
                ],
            }
            await app(scope, receive, send)
            body = b"".join(body_chunks).decode("utf-8", errors="replace")
            return status_code, chunks_read, json.loads(body)

        status, chunks_read, data = asyncio.run(run_chunked_under())
        self.assertEqual(status, 200)
        self.assertEqual(chunks_read, len(chunks))
        self.assertTrue(data.get("success"))

    def test_chunked_streaming_oversized_aborts_early_and_returns_413(self):
        """Chunked stream exceeding byte limit without Content-Length aborts early with HTTP 413."""
        import asyncio
        import json

        boundary = "----TestBoundaryOversized"
        part_header = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"workspace_id\"\r\n\r\n"
            f"1\r\n"
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"test_oversized.csv\"\r\n"
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8")

        chunk_data = b"col1,col2\n" + (b"A" * 65520) + b"\n"
        chunk_footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        # 42 chunks * 64 KiB ~= 2.75 MB
        chunks = [part_header] + [chunk_data for _ in range(40)] + [chunk_footer]

        async def run_chunked_over():
            chunks_read = 0
            idx = 0

            async def receive():
                nonlocal chunks_read, idx
                if idx >= len(chunks):
                    return {"type": "http.request", "body": b"", "more_body": False}
                chunk = chunks[idx]
                idx += 1
                chunks_read += 1
                return {"type": "http.request", "body": chunk, "more_body": (idx < len(chunks))}

            status_code = None
            body_chunks = []
            async def send(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                elif message["type"] == "http.response.body":
                    body_chunks.append(message.get("body", b""))

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/analytics/sources/upload",
                "query_string": b"",
                "headers": [
                    (b"host", b"testserver"),
                    (b"x-api-key", self.test_key.encode("utf-8")),
                    (b"content-type", f"multipart/form-data; boundary={boundary}".encode("utf-8")),
                ],
            }
            # Limit: 64 KiB for analytics (+ 1 MiB allowance in middleware = 1,114,112 bytes)
            with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", 64 * 1024):
                await app(scope, receive, send)

            body = b"".join(body_chunks).decode("utf-8", errors="replace")
            return status_code, chunks_read, json.loads(body)

        status, chunks_read, data = asyncio.run(run_chunked_over())
        self.assertEqual(status, 413)
        self.assertEqual(data.get("detail"), "Request too large")
        # 18 chunks of ~64 KiB = ~1.17 MB crosses the 1.06 MB threshold
        self.assertEqual(
            chunks_read,
            18,
            f"Expected streaming early abort at chunk 18, but read {chunks_read} chunks",
        )
        self.assertLess(chunks_read, len(chunks))

    def test_forged_small_content_length_with_oversized_stream_returns_413(self):
        """Forged small Content-Length header is caught by incremental streaming counter."""
        import asyncio
        import json

        boundary = "----TestBoundaryForged"
        part_header = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"workspace_id\"\r\n\r\n"
            f"1\r\n"
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"test_forged.csv\"\r\n"
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8")

        chunk_data = b"col1,col2\n" + (b"A" * 65520) + b"\n"
        chunk_footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        chunks = [part_header] + [chunk_data for _ in range(40)] + [chunk_footer]

        async def run_forged_stream():
            chunks_read = 0
            idx = 0

            async def receive():
                nonlocal chunks_read, idx
                if idx >= len(chunks):
                    return {"type": "http.request", "body": b"", "more_body": False}
                chunk = chunks[idx]
                idx += 1
                chunks_read += 1
                return {"type": "http.request", "body": chunk, "more_body": (idx < len(chunks))}

            status_code = None
            body_chunks = []
            async def send(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                elif message["type"] == "http.response.body":
                    body_chunks.append(message.get("body", b""))

            scope = {
                "type": "http",
                "method": "POST",
                "path": "/analytics/sources/upload",
                "query_string": b"",
                "headers": [
                    (b"host", b"testserver"),
                    (b"x-api-key", self.test_key.encode("utf-8")),
                    (b"content-length", b"100"),  # Forged small Content-Length
                    (b"content-type", f"multipart/form-data; boundary={boundary}".encode("utf-8")),
                ],
            }
            with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", 64 * 1024):
                await app(scope, receive, send)

            body = b"".join(body_chunks).decode("utf-8", errors="replace")
            return status_code, chunks_read, json.loads(body)

        status, chunks_read, data = asyncio.run(run_forged_stream())
        self.assertEqual(status, 413)
        self.assertEqual(data.get("detail"), "Request too large")
        self.assertEqual(chunks_read, 18)

    def test_valid_upload_near_configured_limit_succeeds(self):
        """Upload file size at ~98% of the configured limit succeeds with HTTP 200."""
        # 50 KiB limit
        limit = 50 * 1024
        row = b"ItemABC,12345,Description_Text_Here\n"
        rows_needed = (49 * 1024) // len(row)
        csv_content = b"col1,col2,col3\n" + (row * rows_needed)

        # Confirm test file is near 98% of limit
        self.assertGreater(len(csv_content), limit * 0.95)
        self.assertLess(len(csv_content), limit)

        with patch.object(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", limit):
            response = self.client.post(
                "/analytics/sources/upload",
                files={"file": ("near_limit.csv", io.BytesIO(csv_content), "text/csv")},
                data={"workspace_id": 1},
                headers=self.auth_headers,
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("filename"), "near_limit.csv")


if __name__ == "__main__":
    unittest.main()
