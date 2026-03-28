"""Session manager for multi-session agent execution."""

from __future__ import annotations

from pathlib import Path

from underthesea.agent.harness.context import ContextManager, HandoffData
from underthesea.agent.harness.progress import ProgressTracker, SubtaskStatus


class SessionManager:
    """Manages multi-session agent execution with progress tracking and context handoff.

    Implements the Anthropic pattern: each session starts with a clean context
    (agent.reset()) and receives structured handoff data from the previous session,
    rather than accumulating context indefinitely.
    """

    def __init__(
        self,
        agent,
        progress_file: str | Path = "progress.json",
        handoff_dir: str | Path = ".agent_state",
    ):
        """
        Initialize SessionManager.

        Parameters
        ----------
        agent : Agent
            The agent instance to run across sessions.
        progress_file : str or Path
            Path to the JSON file for progress tracking.
        handoff_dir : str or Path
            Directory for storing handoff files between sessions.
        """
        self.agent = agent
        self.progress = ProgressTracker(progress_file)
        self.context = ContextManager(handoff_dir)
        self._session_id: int = 0

    def create_task(self, task: str, subtasks: list[str]) -> None:
        """Initialize a new multi-session task with subtasks."""
        self.progress.create(task, subtasks)
        self._session_id = 0

    def run(self) -> dict:
        """Execute one session: pick next subtask, run it, save state.

        Returns
        -------
        dict
            Session results with keys: status, session_id, subtask_id (if applicable).
        """
        # Load existing state
        handoff = self.context.load_latest_handoff()
        self._session_id = (handoff.session_id + 1) if handoff else 1

        # Context reset (clean slate per Anthropic recommendation)
        self.agent.reset()

        # Build context from previous handoff
        original_instruction = self.agent.instruction
        if handoff:
            context_prompt = self.context.build_context_prompt(handoff)
            self.agent.instruction = f"{original_instruction}\n\n{context_prompt}"

        try:
            return self._execute_subtask()
        finally:
            # Restore original instruction
            self.agent.instruction = original_instruction

    def _execute_subtask(self) -> dict:
        """Find and execute the next pending subtask."""
        progress_data = self.progress.load()
        subtask = self.progress.next_pending()
        if subtask is None:
            return {"status": "complete", "message": "All subtasks completed"}

        # Mark in progress
        self.progress.update_subtask(subtask.id, SubtaskStatus.IN_PROGRESS)

        # Execute
        prompt = (
            f"Task: {progress_data.task}\n"
            f"Current subtask: {subtask.description}\n"
            f"Please complete this subtask."
        )
        try:
            response = self.agent(prompt)
            self.progress.update_subtask(
                subtask.id, SubtaskStatus.COMPLETED, result_summary=response[:500]
            )
            status = "completed"
        except Exception as e:
            response = str(e)
            self.progress.update_subtask(
                subtask.id, SubtaskStatus.FAILED, result_summary=response[:500]
            )
            status = "failed"

        # Save handoff for next session
        new_handoff = HandoffData(
            session_id=self._session_id,
            previous_session_summary=response[:500] if response else "",
            current_subtask_id=subtask.id,
            context_for_next_session="Continue with next pending subtask.",
        )
        self.context.save_handoff(new_handoff)

        return {
            "status": status,
            "session_id": self._session_id,
            "subtask_id": subtask.id,
        }

    def run_until_complete(self, max_sessions: int = 10) -> list[dict]:
        """Run sessions until all subtasks complete or max_sessions reached.

        Parameters
        ----------
        max_sessions : int
            Maximum number of sessions to run.

        Returns
        -------
        list[dict]
            Results from each session.
        """
        results = []
        for _ in range(max_sessions):
            result = self.run()
            results.append(result)
            if result["status"] == "complete":
                break
            if self.progress.is_complete():
                break
        return results
