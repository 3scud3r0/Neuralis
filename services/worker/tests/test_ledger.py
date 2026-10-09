from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from neuralis_worker.ledger import Ledger, LedgerError
from neuralis_worker.protocol import Lease


class LedgerTests(unittest.TestCase):
    def test_completion_survives_restart_and_outbox_ack(self):
        with TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory))
            lease = Lease("lease-a", "job-a", {"task": "xor"})
            self.assertTrue(ledger.reserve(lease))
            ledger.complete(lease, "completed", {"loss": .02}, None)
            fresh = Ledger(Path(directory))
            pending = fresh.pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0][2], {"loss": .02})
            fresh.mark_delivery_attempt(lease.lease_id)
            fresh.delivered(lease.lease_id)
            self.assertEqual(fresh.pending(), [])
            self.assertFalse(fresh.reserve(lease))

    def test_interrupted_lease_is_failed_without_reexecution(self):
        with TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory))
            lease = Lease("lease-a", "job-a", {"task": "xor"})
            ledger.reserve(lease)
            self.assertEqual(ledger.recover(), 1)
            pending = ledger.pending()[0]
            self.assertEqual(pending[1], "failed")
            self.assertIn("not repeated", pending[3])
            self.assertFalse(ledger.reserve(lease))
            self.assertEqual(ledger.recover(), 0)

    def test_reused_lease_with_different_config_is_refused(self):
        with TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory))
            ledger.reserve(Lease("lease-a", "job-a", {"task": "xor"}))
            with self.assertRaises(LedgerError):
                ledger.reserve(Lease("lease-a", "job-b", {"task": "xor"}))

    def test_oversized_result_has_bounded_failed_record(self):
        with TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory))
            lease = Lease("lease-a", "job-a", {"task": "xor"})
            ledger.reserve(lease)
            ledger.complete(lease, "completed", {"too_big": "x" * (1024 * 1024)}, None)
            pending = ledger.pending()[0]
            self.assertEqual(pending[1], "failed")
            self.assertEqual(pending[2], {})
            self.assertIn("budget", pending[3])

    def test_history_pruning_preserves_undelivered_completions(self):
        with TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory), max_history=10)
            for i in range(15):
                lease = Lease(f"lease-{i}", f"job-{i}", {"task": "xor"})
                ledger.reserve(lease)
                ledger.complete(lease, "completed", {}, None)
                if i != 0:
                    ledger.delivered(lease.lease_id)
            ledger.prune()
            self.assertEqual(len(ledger.pending()), 1)
            self.assertEqual(ledger.pending()[0][0].lease_id, "lease-0")
            self.assertLessEqual(sum(ledger.stats()["states"].values()), 11)


if __name__ == "__main__":
    unittest.main()
