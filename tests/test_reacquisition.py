import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_continuous_contract as contracts
Adapter = contracts.Adapter
from segmentation.continuous import ContinuousTracker


class ReacquisitionTests(unittest.TestCase):
    fixture = contracts.ContinuousTests.fixture

    def test_long_occlusion_recovers_same_id_without_hidden_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            frames, prompts = self.fixture(Path(folder), count=50,
                occlusion=lambda oid, i: oid == 1 and 5 <= i < 30)
            adapter = Adapter()
            tracker = ContinuousTracker(adapter, window_frames=8, recovery_frames=12,
                                        enable_reacquisition=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rows = list(tracker.predict(frames, prompts))
            self.assertEqual([i for i, _ in rows], list(range(50)))
            self.assertTrue(all(not rows[i][1][1].any() for i in range(5, 30)))
            self.assertTrue(all(rows[i][1][1].any() for i in range(30, 50)))
            self.assertEqual(tracker.lost_object_ids, [])
            self.assertEqual(tracker.ever_lost_object_ids, [1])
            self.assertEqual(tracker.reacquired_object_ids, [1])
            self.assertTrue(all(n <= 23 for n, _, _ in adapter.calls))
            self.assertFalse(list(Path(folder).glob('continuous-*')))

    def test_all_lost_keep_only_compact_actual_reference_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            frames, prompts = self.fixture(Path(folder), count=50,
                occlusion=lambda oid, i: 5 <= i < 30)
            adapter = Adapter()
            tracker = ContinuousTracker(adapter, window_frames=8, recovery_frames=12,
                                        enable_reacquisition=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rows = list(tracker.predict(frames, prompts))
            self.assertEqual(len(rows), 50)
            self.assertTrue(all(not m.any() for i in range(5, 30) for m in rows[i][1].values()))
            self.assertTrue(all(m.any() for i in range(30, 50) for m in rows[i][1].values()))
            self.assertEqual(set(tracker.reacquired_object_ids), {1, 2, 3})
            self.assertTrue(all(n <= 23 for n, _, _ in adapter.calls))
            self.assertTrue(any(n <= 9 and seeds for n, seeds, _ in adapter.calls))

    def test_recovery_cannot_claim_visible_other_target_even_if_colors_match(self):
        class DuplicateAdapter(Adapter):
            def predict(self, frames, prompts, **kwargs):
                for i, masks in super().predict(frames, prompts, **kwargs):
                    if 1 in masks and not masks[1].any():
                        masks[1] = masks[2].copy()
                    yield i, masks
        with tempfile.TemporaryDirectory() as folder:
            frames, prompts = self.fixture(Path(folder), count=40,
                occlusion=lambda oid, i: oid == 1 and i >= 5)
            tracker = ContinuousTracker(DuplicateAdapter(), window_frames=8,
                                        recovery_frames=12, enable_reacquisition=True)
            # Isolate overlap protection in the worst case of indistinguishable colors.
            with patch('segmentation.continuous.AppearanceMemory.matches', return_value=True), \
                 contextlib.redirect_stdout(io.StringIO()):
                rows = list(tracker.predict(frames, prompts))
            self.assertTrue(all(not masks[1].any() for i, masks in rows if i >= 5))
            self.assertTrue(all(masks[2].any() for _, masks in rows))
            self.assertEqual(tracker.lost_object_ids, [1])

    def test_reacquisition_error_removes_compact_window(self):
        with tempfile.TemporaryDirectory() as folder:
            frames, prompts = self.fixture(Path(folder), count=10)
            adapter = Adapter('shape')
            tracker = ContinuousTracker(adapter, enable_reacquisition=True)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                list(tracker.predict(frames, prompts))
            self.assertEqual(adapter.closed, 1)
            self.assertFalse(list(Path(folder).glob('continuous-*')))
