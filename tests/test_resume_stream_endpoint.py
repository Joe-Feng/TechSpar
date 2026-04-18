import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from backend.models import ChatRequest
from backend.routers.interview import chat_stream


class _FakeState:
    def __init__(self, next_value, values=None):
        self.next = next_value
        self.values = values or {}


class _FakeGraph:
    def __init__(self, reply: str):
        self.reply = reply
        self.invoke_called = False
        self.updated_messages = []
        self._state_calls = 0

    def get_state(self, _config):
        self._state_calls += 1
        if self._state_calls == 1:
            return _FakeState(["wait"])
        return _FakeState(["wait"], {"is_finished": False, "phase": "self_intro"})

    def update_state(self, _config, payload):
        self.updated_messages.append(payload)

    def invoke(self, _input, _config):
        self.invoke_called = True
        return {
            "messages": [AIMessage(content=self.reply)],
            "is_finished": False,
            "phase": "self_intro",
        }

    async def astream_events(self, *_args, **_kwargs):
        raise AssertionError("chat_stream should not use astream_events for resume mode")


class ResumeStreamEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_stream_reuses_sync_invoke_and_emits_sse_tokens(self):
        graph = _FakeGraph("继续说说你在项目里的职责。")
        entry = {
            "graph": graph,
            "config": {"configurable": {"thread_id": "session-1"}},
            "user_id": "user-1",
        }

        with patch("backend.routers.interview.get_or_restore_resume_graph", return_value=entry), patch(
            "backend.routers.interview.append_message"
        ) as append_message:
            response = await chat_stream(ChatRequest(session_id="session-1", message="你好"), user_id="user-1")
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)

        body = "".join(chunks)
        events = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        tokens = "".join(event.get("token", "") for event in events)

        self.assertTrue(graph.invoke_called)
        self.assertEqual(tokens, "继续说说你在项目里的职责。")
        self.assertTrue(events[-1]["done"])
        self.assertFalse(events[-1]["is_finished"])
        self.assertNotIn("error", body)
        append_message.assert_any_call("session-1", "user", "你好", user_id="user-1")
        append_message.assert_any_call("session-1", "assistant", "继续说说你在项目里的职责。", user_id="user-1")


if __name__ == "__main__":
    unittest.main()
