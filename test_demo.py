import unittest

from demo import TemporaryFailure, build_report, context_check, verify, with_fallback


class DemoTests(unittest.TestCase):
    def test_three_way_verdict(self):
        self.assertEqual(verify({"observed": 7, "claimed": 7}), "S")
        self.assertEqual(verify({"observed": 7, "claimed": 3}), "C")
        self.assertEqual(verify({"observed": None, "claimed": 7}), "U")

    def test_retry_limit_and_fallback(self):
        result = build_report()["resilience"]
        self.assertEqual(result["judge_model"], "synthetic-backup")
        self.assertEqual(result["planned_waits_s"], [0.25, 0.5, 1.0])
        self.assertEqual(result["events"][-1]["event"], "fallback")

    def test_permanent_error(self):
        def unauthorized():
            raise TemporaryFailure(401)
        with self.assertRaises(TemporaryFailure):
            with_fallback(unauthorized, lambda: self.fail("Unexpected fallback"), wait=lambda _: None)

    def test_context_boundary_counts_bytes(self):
        self.assertEqual(context_check("证据", 6)["state"], "ready")
        self.assertEqual(context_check("证据", 5)["state"], "context_overflow")


if __name__ == "__main__":
    unittest.main()
