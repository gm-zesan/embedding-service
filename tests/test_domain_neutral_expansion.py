import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

from app import config
from app.llm.models import LLMResponse
from app.retrieval_engine import expand_query_via_llm


class TestDomainNeutralExpansion(unittest.TestCase):
    """
    Verification suite for IMP-DOM-003: Retrieval Expansion Neutrality.
    Ensures Tier 3 LLM query expansion is domain-neutral across healthcare,
    education, hospitality, SaaS, and commerce without hardcoded retail bias.
    """

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_expansion_disabled_when_api_key_missing(self):
        """When LLM_EXPANSION_API_KEY is not configured, expansion returns None immediately."""
        with patch.object(config, "LLM_EXPANSION_API_KEY", ""):
            result = self.loop.run_until_complete(expand_query_via_llm("ডাক্তার কখন বসবেন?"))
            self.assertNull_or_None = self.assertIsNone(result)

    def test_expansion_prompt_is_domain_neutral(self):
        """
        Verify the system prompt contains domain-neutral instructions
        and does NOT constrain topics exclusively to retail FAQs (refund, warranty, delivery charges).
        """
        captured_requests = []

        async def fake_generate(req):
            captured_requests.append(req)
            return LLMResponse(
                content="doctor availability, appointment schedule, সিরিয়াল, ডাক্তার",
                provider="mock",
                model="mock-v1",
            )

        with patch.object(config, "LLM_EXPANSION_API_KEY", "test-key"), \
             patch("app.retrieval_engine.llm_client.generate", side_effect=fake_generate):

            result = self.loop.run_until_complete(expand_query_via_llm("daktar kobe thakben?"))

            self.assertIsNotNone(result)
            self.assertEqual(len(captured_requests), 1)

            req = captured_requests[0]
            system_msg = next((m["content"] for m in req.messages if m["role"] == "system"), "")

            # Verify domain neutrality: Must NOT have the old retail-only lock-in
            self.assertNotIn(
                "(e.g. refund, cancellation, delivery charges, payment methods, warranty)",
                system_msg,
                "Prompt must not restrict topics to retail operations only"
            )

            # Must mention multi-domain coverage and domain neutrality
            self.assertIn("domain-neutral", system_msg.lower())
            self.assertIn("healthcare", system_msg.lower())
            self.assertIn("education", system_msg.lower())
            self.assertIn("hospitality", system_msg.lower())

            # Verify prompt contains multi-domain examples
            self.assertIn("daktar kobe thakben?", system_msg)
            self.assertIn("admission fee koto?", system_msg)
            self.assertIn("order kobe pabo?", system_msg)
            self.assertIn("room booking cancel kora jabe?", system_msg)

    def test_expansion_across_multiple_domains(self):
        """
        Verify expand_query_via_llm faithfully passes domain-specific queries
        and returns clean comma-separated output.
        """
        test_cases = [
            (
                "healthcare",
                "icu bed faka ache kina?",
                "icu bed availability, hospital admission, icu status, আইসিইউ বেড",
            ),
            (
                "education",
                "admission requirements ki ki?",
                "admission eligibility, entry criteria, requirements, ভর্তি যোগ্যতা",
            ),
            (
                "hospitality",
                "hotel check in time koto?",
                "check-in time, hotel policy, arrival schedule, চেক ইন সময়",
            ),
            (
                "saas_it",
                "api key kivabe generate korbo?",
                "api key creation, token generate, developer settings, এপিআই কি",
            ),
            (
                "retail",
                "delivery charge koto?",
                "delivery fee, shipping cost, delivery charges, ডেলিভারি চার্জ",
            ),
        ]

        for domain, query, expected_output in test_cases:
            with self.subTest(domain=domain, query=query):
                async def fake_generate(req):
                    return LLMResponse(
                        content=expected_output,
                        provider="mock",
                        model="mock-v1",
                    )

                with patch.object(config, "LLM_EXPANSION_API_KEY", "test-key"), \
                     patch("app.retrieval_engine.llm_client.generate", side_effect=fake_generate):

                    result = self.loop.run_until_complete(expand_query_via_llm(query))
                    self.assertEqual(result, expected_output)

    def test_expansion_handles_llm_exception_gracefully(self):
        """When LLM provider raises an exception, expansion safely catches and returns None."""
        async def failing_generate(req):
            raise RuntimeError("LLM Provider Timeout / Rate Limit")

        with patch.object(config, "LLM_EXPANSION_API_KEY", "test-key"), \
             patch("app.retrieval_engine.llm_client.generate", side_effect=failing_generate):

            result = self.loop.run_until_complete(expand_query_via_llm("test query"))
            self.assertIsNone(result)

    def test_expansion_handles_empty_response(self):
        """When LLM returns an empty content response, expansion safely returns None."""
        async def empty_generate(req):
            return LLMResponse(
                content="",
                provider="mock",
                model="mock-v1",
            )

        with patch.object(config, "LLM_EXPANSION_API_KEY", "test-key"), \
             patch("app.retrieval_engine.llm_client.generate", side_effect=empty_generate):

            result = self.loop.run_until_complete(expand_query_via_llm("test query"))
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

