"""Tests for agent harness (progress, context, session)."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

from underthesea.agent.harness.context import ContextManager, HandoffData
from underthesea.agent.harness.progress import ProgressTracker, SubtaskStatus


class TestProgressTracker(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.progress_file = Path(self.tmp_dir) / "progress.json"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_create_progress(self):
        tracker = ProgressTracker(self.progress_file)
        data = tracker.create("Analyze documents", ["Read docs", "Summarize", "Report"])

        self.assertEqual(data.task, "Analyze documents")
        self.assertEqual(len(data.subtasks), 3)
        self.assertEqual(data.subtasks[0].id, 1)
        self.assertEqual(data.subtasks[0].description, "Read docs")
        self.assertEqual(data.subtasks[0].status, SubtaskStatus.PENDING)
        self.assertTrue(self.progress_file.exists())

    def test_load_progress(self):
        tracker = ProgressTracker(self.progress_file)
        tracker.create("Test task", ["Step 1", "Step 2"])

        tracker2 = ProgressTracker(self.progress_file)
        data = tracker2.load()

        self.assertEqual(data.task, "Test task")
        self.assertEqual(len(data.subtasks), 2)

    def test_update_subtask_status(self):
        tracker = ProgressTracker(self.progress_file)
        tracker.create("Task", ["Sub 1", "Sub 2", "Sub 3"])

        tracker.update_subtask(1, SubtaskStatus.IN_PROGRESS)
        data = tracker.load()
        self.assertEqual(data.subtasks[0].status, SubtaskStatus.IN_PROGRESS)
        self.assertIsNotNone(data.subtasks[0].started_at)

        tracker.update_subtask(1, SubtaskStatus.COMPLETED, result_summary="Done!")
        data = tracker.load()
        self.assertEqual(data.subtasks[0].status, SubtaskStatus.COMPLETED)
        self.assertEqual(data.subtasks[0].result_summary, "Done!")
        self.assertIsNotNone(data.subtasks[0].completed_at)

    def test_next_pending(self):
        tracker = ProgressTracker(self.progress_file)
        tracker.create("Task", ["A", "B", "C"])

        subtask = tracker.next_pending()
        self.assertEqual(subtask.id, 1)

        tracker.update_subtask(1, SubtaskStatus.COMPLETED)
        subtask = tracker.next_pending()
        self.assertEqual(subtask.id, 2)

        tracker.update_subtask(2, SubtaskStatus.COMPLETED)
        tracker.update_subtask(3, SubtaskStatus.COMPLETED)
        subtask = tracker.next_pending()
        self.assertIsNone(subtask)

    def test_is_complete(self):
        tracker = ProgressTracker(self.progress_file)
        tracker.create("Task", ["A", "B"])

        self.assertFalse(tracker.is_complete())

        tracker.update_subtask(1, SubtaskStatus.COMPLETED)
        self.assertFalse(tracker.is_complete())

        tracker.update_subtask(2, SubtaskStatus.SKIPPED)
        self.assertTrue(tracker.is_complete())

    def test_failed_subtask_not_complete(self):
        tracker = ProgressTracker(self.progress_file)
        tracker.create("Task", ["A"])
        tracker.update_subtask(1, SubtaskStatus.FAILED)

        self.assertFalse(tracker.is_complete())

    def test_json_persistence(self):
        tracker = ProgressTracker(self.progress_file)
        tracker.create("Persistence test", ["Step 1"])
        tracker.update_subtask(1, SubtaskStatus.COMPLETED, result_summary="OK")

        with open(self.progress_file) as f:
            raw = json.load(f)

        self.assertEqual(raw["task"], "Persistence test")
        self.assertEqual(raw["subtasks"][0]["status"], "completed")
        self.assertEqual(raw["subtasks"][0]["result_summary"], "OK")


class TestContextManager(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_save_and_load_handoff(self):
        ctx = ContextManager(self.tmp_dir)
        handoff = HandoffData(
            session_id=1,
            previous_session_summary="Completed step 1",
            current_subtask_id=1,
            context_for_next_session="Continue with step 2",
            artifacts=["/tmp/output.txt"],
            warnings="File #3 has encoding issues",
        )

        path = ctx.save_handoff(handoff)
        self.assertTrue(path.exists())

        loaded = ctx.load_handoff(1)
        self.assertEqual(loaded.session_id, 1)
        self.assertEqual(loaded.previous_session_summary, "Completed step 1")
        self.assertEqual(loaded.current_subtask_id, 1)
        self.assertEqual(loaded.artifacts, ["/tmp/output.txt"])
        self.assertEqual(loaded.warnings, "File #3 has encoding issues")

    def test_load_latest_handoff(self):
        ctx = ContextManager(self.tmp_dir)
        ctx.save_handoff(HandoffData(session_id=1, previous_session_summary="Session 1"))
        ctx.save_handoff(HandoffData(session_id=2, previous_session_summary="Session 2"))
        ctx.save_handoff(HandoffData(session_id=3, previous_session_summary="Session 3"))

        latest = ctx.load_latest_handoff()
        self.assertEqual(latest.session_id, 3)
        self.assertEqual(latest.previous_session_summary, "Session 3")

    def test_load_latest_no_handoffs(self):
        ctx = ContextManager(self.tmp_dir)
        self.assertIsNone(ctx.load_latest_handoff())

    def test_load_nonexistent_session(self):
        ctx = ContextManager(self.tmp_dir)
        self.assertIsNone(ctx.load_handoff(999))

    def test_build_context_prompt(self):
        ctx = ContextManager(self.tmp_dir)
        handoff = HandoffData(
            session_id=2,
            previous_session_summary="Analyzed 50 documents",
            context_for_next_session="Focus on financial docs next",
            artifacts=["/tmp/analysis.json", "/tmp/summary.md"],
            warnings="Document #15 corrupted",
        )

        prompt = ctx.build_context_prompt(handoff)

        self.assertIn("Session 2", prompt)
        self.assertIn("Analyzed 50 documents", prompt)
        self.assertIn("Focus on financial docs next", prompt)
        self.assertIn("/tmp/analysis.json", prompt)
        self.assertIn("Document #15 corrupted", prompt)


class TestSessionManager(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.progress_file = Path(self.tmp_dir) / "progress.json"
        self.handoff_dir = Path(self.tmp_dir) / "handoffs"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _make_mock_agent(self, responses):
        """Create a mock agent that returns responses in sequence."""
        agent = Mock()
        agent.instruction = "You are a helpful assistant."
        agent.reset = Mock()
        agent.side_effect = responses
        return agent

    def test_create_task(self):
        from underthesea.agent.harness import SessionManager

        agent = self._make_mock_agent([])
        sm = SessionManager(agent, self.progress_file, self.handoff_dir)
        sm.create_task("Big task", ["Step 1", "Step 2", "Step 3"])

        self.assertTrue(self.progress_file.exists())
        data = sm.progress.load()
        self.assertEqual(data.task, "Big task")
        self.assertEqual(len(data.subtasks), 3)

    def test_run_single_session(self):
        from underthesea.agent.harness import SessionManager

        agent = self._make_mock_agent(["Result for step 1"])
        sm = SessionManager(agent, self.progress_file, self.handoff_dir)
        sm.create_task("Task", ["Step 1", "Step 2"])

        result = sm.run()

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["session_id"], 1)
        self.assertEqual(result["subtask_id"], 1)
        agent.reset.assert_called_once()

    def test_run_until_complete(self):
        from underthesea.agent.harness import SessionManager

        agent = self._make_mock_agent(["Done 1", "Done 2"])
        sm = SessionManager(agent, self.progress_file, self.handoff_dir)
        sm.create_task("Task", ["Step 1", "Step 2"])

        results = sm.run_until_complete(max_sessions=5)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(results[1]["status"], "completed")
        self.assertTrue(sm.progress.is_complete())

    def test_context_reset_between_sessions(self):
        from underthesea.agent.harness import SessionManager

        agent = self._make_mock_agent(["Done 1", "Done 2"])
        sm = SessionManager(agent, self.progress_file, self.handoff_dir)
        sm.create_task("Task", ["Step 1", "Step 2"])

        sm.run_until_complete()

        # Agent should be reset at the start of each session
        self.assertEqual(agent.reset.call_count, 2)

    def test_handoff_created_between_sessions(self):
        from underthesea.agent.harness import SessionManager

        agent = self._make_mock_agent(["Done 1", "Done 2"])
        sm = SessionManager(agent, self.progress_file, self.handoff_dir)
        sm.create_task("Task", ["Step 1", "Step 2"])

        sm.run_until_complete()

        # Handoff files should exist
        handoff_files = list(self.handoff_dir.glob("handoff_session_*.json"))
        self.assertEqual(len(handoff_files), 2)

    def test_all_complete_returns_complete(self):
        from underthesea.agent.harness import SessionManager

        agent = self._make_mock_agent(["Done"])
        sm = SessionManager(agent, self.progress_file, self.handoff_dir)
        sm.create_task("Task", ["Only step"])

        sm.run()
        result = sm.run()

        self.assertEqual(result["status"], "complete")

    def test_failed_subtask(self):
        from underthesea.agent.harness import SessionManager

        agent = Mock()
        agent.instruction = "Test"
        agent.reset = Mock()
        agent.side_effect = RuntimeError("LLM error")

        sm = SessionManager(agent, self.progress_file, self.handoff_dir)
        sm.create_task("Task", ["Step 1"])

        result = sm.run()

        self.assertEqual(result["status"], "failed")
        data = sm.progress.load()
        self.assertEqual(data.subtasks[0].status, SubtaskStatus.FAILED)

    def test_instruction_restored_after_session(self):
        from underthesea.agent.harness import SessionManager

        agent = self._make_mock_agent(["Done 1", "Done 2"])
        original_instruction = "Original instruction"
        agent.instruction = original_instruction

        sm = SessionManager(agent, self.progress_file, self.handoff_dir)
        sm.create_task("Task", ["Step 1", "Step 2"])

        sm.run_until_complete()

        self.assertEqual(agent.instruction, original_instruction)
