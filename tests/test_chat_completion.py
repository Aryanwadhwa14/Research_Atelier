"""Offline regressions for chatbot completion and output-limit handling."""

import unittest
from typing import ClassVar
from unittest.mock import Mock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

import agent


class ScriptedChatModel(BaseChatModel):
    """Return predetermined tool calls and a final model message."""

    responses: ClassVar[list] = []
    calls: ClassVar[int] = 0
    observed_messages: ClassVar[list] = []

    @property
    def _llm_type(self):
        return "scripted-chat-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).observed_messages.append(list(messages))
        response = type(self).responses[type(self).calls]
        type(self).calls += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class ChatCompletionTests(unittest.TestCase):
    def make_graph(self, responses, tool_function):
        ScriptedChatModel.responses = responses
        ScriptedChatModel.calls = 0
        ScriptedChatModel.observed_messages = []
        return create_react_agent(
            model=ScriptedChatModel(),
            tools=[tool_function],
            prompt="Use tools when needed, then answer the user.",
            checkpointer=MemorySaver(),
            pre_model_hook=agent._close_incomplete_tool_calls,
            post_model_hook=agent._stop_repeated_tool_batch,
        )

    def test_truncated_final_after_tool_uses_one_bounded_completion_attempt(self):
        executed = []

        @tool
        def vector_db_lookup(query: str) -> str:
            """Return controlled comparison evidence."""
            executed.append(query)
            return "For vector databases, Milvus scales well but needs operational expertise. Qdrant offers a simpler managed experience. [Milvus docs](https://example.org/milvus)"

        first = AIMessage(content="", tool_calls=[{
            "name": "vector_db_lookup", "args": {"query": "vector database trade-offs"},
            "id": "lookup-1", "type": "tool_call",
        }])
        partial = AIMessage(
            content="Milvus offers scale but requires ops effort, complex",
            response_metadata={"finish_reason": "length"},
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": agent.CHAT_MODEL_MAX_TOKENS,
                "total_tokens": 100 + agent.CHAT_MODEL_MAX_TOKENS,
            },
        )
        graph = self.make_graph([first, partial], vector_db_lookup)
        final = AIMessage(content="Milvus offers scale but needs more operations. Qdrant is simpler to run.")
        model = Mock()
        model.invoke.return_value = final

        with patch("agent.agent", graph), patch("agent.llm", model):
            result = agent.run_agent("Compare vector databases", "truncated-final-test")

        self.assertEqual(executed, ["vector database trade-offs"])
        self.assertIn(final.content, result["answer"])
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(model.invoke.call_count, 1)
        self.assertIn("Milvus scales well", model.invoke.call_args.args[0])
        self.assertEqual(model.invoke.call_args.kwargs["max_tokens"], agent.CHAT_FINAL_ATTEMPT_MAX_TOKENS)
        self.assertLessEqual(len(model.invoke.call_args.args[0]), agent.CHAT_FINAL_ATTEMPT_MAX_CHARS)
        self.assertIn("**Sources**", result["answer"])
        self.assertIn("[Milvus docs](https://example.org/milvus)", result["answer"])

    def test_failed_bounded_completion_preserves_partial_with_warning(self):
        class EmptyGraph:
            def invoke(self, values, config):
                return {"messages": [
                    # The current request is the last user message.
                    HumanMessage(content="Compare stores"),
                    AIMessage(content="Milvus requires ops effort, complex"),
                    AIMessage(content="", response_metadata={"finish_reason": "length"}),
                ]}

            def get_state(self, config):
                from types import SimpleNamespace
                return SimpleNamespace(values={"messages": []})

        model = Mock()
        model.invoke.side_effect = RuntimeError("simulated model failure")
        with patch("agent.agent", EmptyGraph()), patch("agent.llm", model):
            result = agent.run_agent("Compare stores", "failed-final-test")

        self.assertTrue(result["answer"].startswith(agent.INCOMPLETE_RESPONSE_PREFIX))
        self.assertIn("Milvus requires ops effort, complex", result["answer"])
        self.assertIn("may be incomplete", result["answer"])
        self.assertNotIn("https://", result["answer"])

    def test_truncated_bounded_completion_preserves_its_partial_with_warning(self):
        class EmptyGraph:
            def invoke(self, values, config):
                return {"messages": [
                    HumanMessage(content="Compare stores"),
                    AIMessage(content="Original draft"),
                    AIMessage(content="", response_metadata={"finish_reason": "length"}),
                ]}

            def get_state(self, config):
                from types import SimpleNamespace
                return SimpleNamespace(values={"messages": []})

        model = Mock()
        model.invoke.return_value = AIMessage(
            content="Milvus scales well, while Qdrant is simpler to operate.",
            response_metadata={"finish_reason": "length"},
        )
        with patch("agent.agent", EmptyGraph()), patch("agent.llm", model):
            result = agent.run_agent("Compare stores", "truncated-final-retry-test")

        self.assertTrue(result["answer"].startswith(agent.INCOMPLETE_RESPONSE_PREFIX))
        self.assertIn("Milvus scales well", result["answer"])
        self.assertNotIn("No complete answer was produced", result["answer"])

    def test_repeated_tool_call_is_not_executed_twice_and_gets_final_attempt(self):
        executed = []

        @tool
        def lookup(query: str) -> str:
            """Return one controlled source result."""
            executed.append(query)
            return "Source says the systems trade off operational complexity and scale."

        first = AIMessage(content="", tool_calls=[{
            "name": "lookup", "args": {"query": "same query"}, "id": "repeat-1", "type": "tool_call",
        }])
        repeated = AIMessage(content="", tool_calls=[{
            "name": "lookup", "args": {"query": "same query"}, "id": "repeat-2", "type": "tool_call",
        }])
        graph = self.make_graph([first, repeated], lookup)
        model = Mock()
        model.invoke.return_value = AIMessage(content="The evidence indicates a scale versus operational simplicity trade-off.")

        with patch("agent.agent", graph), patch("agent.llm", model):
            result = agent.run_agent("Compare vector databases", "repeat-call-test")

        self.assertEqual(executed, ["same query"])
        self.assertEqual(result["answer"], model.invoke.return_value.content)
        self.assertEqual(model.invoke.call_count, 1)
        self.assertEqual(len(result["steps"]), 1)

    def test_truncation_detection_uses_finish_reason_and_token_count(self):
        self.assertIn("compact table", agent.SYSTEM_PROMPT)
        self.assertIn("practical recommendation", agent.SYSTEM_PROMPT)
        self.assertTrue(agent._message_was_truncated(AIMessage(
            content="partial", response_metadata={"finish_reason": "length"}
        )))
        self.assertFalse(agent._message_was_truncated(AIMessage(content="complete")))

    def test_web_timeout_is_nonfatal_when_another_tool_returns_evidence(self):
        from langchain_core.messages import ToolMessage

        @tool
        def local_lookup(topic: str) -> str:
            """Return a useful independent source result."""
            return f"Local evidence for {topic}: Milvus scales, Qdrant is simpler to operate."

        first = AIMessage(content="", tool_calls=[
            {"name": "web_search", "args": {"query": "vector database comparison"}, "id": "web-timeout", "type": "tool_call"},
            {"name": "local_lookup", "args": {"topic": "vector databases"}, "id": "local-evidence", "type": "tool_call"},
        ])
        final = AIMessage(content="Milvus favors scale; Qdrant is simpler to operate, based on the local evidence.")
        # Rebuild with both tools so the real ToolNode handles a timeout and
        # still routes the usable result into the synthesis model.
        ScriptedChatModel.responses = [first, final]
        ScriptedChatModel.calls = 0
        ScriptedChatModel.observed_messages = []
        graph = create_react_agent(
            model=ScriptedChatModel(),
            tools=[agent.web_search, local_lookup],
            prompt=agent.SYSTEM_PROMPT,
            checkpointer=MemorySaver(),
            pre_model_hook=agent._close_incomplete_tool_calls,
            post_model_hook=agent._stop_repeated_tool_batch,
        )
        with patch("agent.DDGS", side_effect=TimeoutError("simulated search timeout")), \
             patch("agent.time.sleep"), patch("agent.agent", graph):
            result = agent.run_agent("Compare vector databases", "search-timeout-with-evidence")

        self.assertIn("Qdrant is simpler", result["answer"])
        final_model_input = ScriptedChatModel.observed_messages[-1]
        tool_messages = [message for message in final_model_input if isinstance(message, ToolMessage)]
        by_id = {message.tool_call_id: message.content for message in tool_messages}
        self.assertIn("Web search is unavailable", by_id["web-timeout"])
        self.assertIn("Local evidence for vector databases", by_id["local-evidence"])


if __name__ == "__main__":
    unittest.main()
