import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from backend.graphs.resume_interview import _make_init_interview, _make_interviewer_ask
from backend.indexer import query_resume
from backend.llm_provider import compat_chat_completion


class _FakeNode:
    def __init__(self, content: str):
        self._content = content

    def get_content(self) -> str:
        return self._content


class _FakeRetriever:
    def __init__(self, contents: list[str]):
        self._contents = contents

    def retrieve(self, _question: str):
        return [_FakeNode(content) for content in self._contents]


class _FakeIndex:
    def __init__(self, contents: list[str]):
        self._contents = contents

    def as_retriever(self, similarity_top_k: int):
        del similarity_top_k
        return _FakeRetriever(self._contents)

    def as_query_engine(self, similarity_top_k: int):
        del similarity_top_k
        raise AssertionError("query_resume should not use LlamaIndex query engine for resume summaries")


class ResumeGatewayCompatTests(unittest.TestCase):
    def test_compat_chat_completion_ignores_empty_messages(self):
        captured = {}

        class _FakeResponse:
            ok = True

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "OK"}}]}

        def fake_post(_url, headers=None, json=None, timeout=None):
            del headers, timeout
            captured["payload"] = json
            return _FakeResponse()

        with patch("backend.llm_provider.requests.post", side_effect=fake_post):
            result = compat_chat_completion(
                [
                    {"role": "assistant", "content": ""},
                    {"role": "user", "content": "你好"},
                ]
            )

        self.assertEqual(result, "OK")
        self.assertEqual(captured["payload"]["messages"], [{"role": "user", "content": "你好"}])

    def test_query_resume_uses_retriever_and_compat_client(self):
        fake_index = _FakeIndex(
            [
                "项目经历：智能座舱多 Agent 语音系统评测",
                "技能：Python、Pytest、FastAPI",
            ]
        )

        with patch("backend.indexer.build_resume_index", return_value=fake_index), patch(
            "backend.indexer.compat_chat_completion",
            return_value="总结后的简历上下文",
            create=True,
        ) as compat_chat:
            result = query_resume("列出候选人的项目经历和技能", "user-1")

        self.assertEqual(result, "总结后的简历上下文")
        compat_chat.assert_called_once()

    def test_init_interview_uses_compat_client_instead_of_langchain_model(self):
        with patch(
            "backend.graphs.resume_interview.query_resume",
            return_value="候选人做过智能座舱多 Agent 评测项目",
        ), patch(
            "backend.graphs.resume_interview.compat_chat_completion",
            return_value="你好，先做个自我介绍吧。",
            create=True,
        ) as compat_chat, patch(
            "backend.graphs.resume_interview.get_langchain_llm",
            side_effect=AssertionError("resume init should not use ChatOpenAI"),
            create=True,
        ):
            result = _make_init_interview("user-1")({})

        self.assertEqual(result["messages"][0].content, "你好，先做个自我介绍吧。")
        compat_chat.assert_called_once()

    def test_init_interview_falls_back_when_gateway_returns_empty_content(self):
        with patch(
            "backend.graphs.resume_interview.query_resume",
            return_value="候选人做过智能座舱多 Agent 评测项目",
        ), patch(
            "backend.graphs.resume_interview.compat_chat_completion",
            return_value="",
            create=True,
        ):
            result = _make_init_interview("user-1")({})

        self.assertTrue(result["messages"][0].content.strip())

    def test_interviewer_ask_uses_compat_client_and_preserves_inline_eval(self):
        state = {
            "resume_context": "候选人熟悉 Python 和 FastAPI",
            "phase": "technical",
            "questions_asked": [],
            "phase_question_count": 0,
            "messages": [HumanMessage(content="我主要做智能座舱评测。")],
        }

        with patch(
            "backend.graphs.resume_interview.compat_chat_completion",
            return_value='你在项目里怎么设计评测维度？\n<!--EVAL:{"score":7,"should_advance":false,"brief":"可以继续追问"}-->',
            create=True,
        ) as compat_chat, patch(
            "backend.graphs.resume_interview.get_langchain_llm",
            side_effect=AssertionError("resume interviewer should not use ChatOpenAI"),
            create=True,
        ):
            result = _make_interviewer_ask("user-1")(state)

        self.assertEqual(result["messages"][0].content, "你在项目里怎么设计评测维度？")
        self.assertEqual(result["last_eval"]["score"], 7)
        self.assertFalse(result["last_eval"]["should_advance"])
        compat_chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
