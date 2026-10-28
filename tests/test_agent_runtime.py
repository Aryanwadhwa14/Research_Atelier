"""Offline regressions for web-search failures and checkpoint recovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

import agent


class WebSearchReliabilityTests(unittest.TestCase):
    def test_groq_configuration_uses_bounded_output_timeout_and_retry(self):
        self.assertEqual(agent.llm.model_name, "openai/gpt-oss-20b")
        self.assertEqual(agent.llm.max_tokens, 900)
        self.assertEqual(agent.llm.max_retries, 1)
        self.assertEqual(agent.WEB_SEARCH_BACKENDS, "auto")
        self.assertEqual(agent.WEB_SEARCH_ATTEMPTS, 1)

    def test_model_metrics_log_duration_and_available_token_counts_without_prompt_text(self):
        callback = agent._ModelTimingCallback()
        response = SimpleNamespace(
            llm_output={"token_usage": {"prompt_tokens": 17, "completion_tokens": 9}},
            generations=[[SimpleNamespace(
                generation_info={"finish_reason": "length"},
                message=SimpleNamespace(usage_metadata={}),
            )]],
        )
        with patch("agent.time.perf_counter", side_effect=[10.0, 10.125]):
            callback.on_chat_model_start({}, [], run_id="metrics-test")
            with self.assertLogs("agent", level="INFO") as captured:
                callback.on_llm_end(response, run_id="metrics-test")

        self.assertIn("duration_ms=125.0", captured.output[0])
        self.assertIn("input_tokens=17", captured.output[0])
        self.assertIn("output_tokens=9", captured.output[0])
        self.assertIn("finish_reason=length", captured.output[0])
        self.assertNotIn("prompt text", captured.output[0])

    def test_timeout_retries_are_bounded_and_return_an_explicit_failure(self):
        attempted_backends = []

        class TimedOutSearch:
            def __init__(self, *, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def text(self, query, *, max_results, backend):
                attempted_backends.append(backend)
                self.query = query
                self.max_results = max_results
                self.backend = backend
                raise TimeoutError("provider request timed out")

        with patch("agent.DDGS", side_effect=TimedOutSearch) as ddgs, patch("agent.time.sleep") as sleep:
            result = agent.web_search.invoke({"query": "recent RAG research"})

        self.assertEqual(ddgs.call_count, agent.WEB_SEARCH_ATTEMPTS)
        self.assertTrue(all(call.kwargs == {"timeout": agent.WEB_SEARCH_TIMEOUT_SECONDS} for call in ddgs.call_args_list))
        self.assertEqual(attempted_backends, ["auto"] * agent.WEB_SEARCH_ATTEMPTS)
        self.assertEqual(sleep.call_count, 0)
        self.assertIn("Web search is unavailable", result)
        self.assertIn("1 attempt(s)", result)
        self.assertIn("No web results were retrieved", result)
        self.assertNotIn("provider request timed out", result)

    def test_failed_web_search_and_another_tool_both_return_tool_messages(self):
        class TimedOutSearch:
            def __init__(self, *, timeout):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def text(self, query, *, max_results, backend):
                raise TimeoutError("offline mocked provider")

        @tool
        def local_lookup(topic: str) -> str:
            """Return a controlled local test result."""
            return f"Local source for {topic}."

        requested_calls = [
            {"name": "web_search", "args": {"query": "recent RAG research"}, "id": "web-1", "type": "tool_call"},
            {"name": "local_lookup", "args": {"topic": "RAG"}, "id": "local-1", "type": "tool_call"},
        ]
        with patch("agent.DDGS", side_effect=TimedOutSearch), patch("agent.time.sleep"):
            builder = StateGraph(MessagesState)
            builder.add_node("tools", ToolNode([agent.web_search, local_lookup]))
            builder.add_edge(START, "tools")
            builder.add_edge("tools", END)
            tool_result = builder.compile().invoke(
                {"messages": [AIMessage(content="", tool_calls=requested_calls)]}
            )

        messages = [message for message in tool_result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual({message.tool_call_id for message in messages}, {"web-1", "local-1"})
        by_id = {message.tool_call_id: message for message in messages}
        self.assertIn("Web search is unavailable", by_id["web-1"].content)
        self.assertIn("Local source for RAG", by_id["local-1"].content)


class IncompleteCheckpointTests(unittest.TestCase):
    def test_chat_history_is_bounded_by_complete_recent_turns(self):
        messages = []
        for index in range(10):
            messages.extend([
                HumanMessage(content=f"old question {index}"),
                AIMessage(content=f"old answer {index}"),
            ])
        messages.append(HumanMessage(content="latest question"))

        bounded, changed = agent._bound_chat_history(messages)

        self.assertTrue(changed)
        self.assertLessEqual(sum(agent._message_size(message) for message in bounded), agent.CHAT_HISTORY_MAX_CHARS)
        self.assertIn("latest question", [message.content for message in bounded])
        self.assertNotIn("old question 0", [message.content for message in bounded])
        self.assertEqual(sum(message.type == "human" for message in bounded), agent.CHAT_HISTORY_MAX_TURNS)

    def test_real_react_agent_repairs_partial_multi_tool_checkpoint_before_model(self):
        """Reproduce a checkpoint saved after only part of web/arXiv tool batch."""
        calls = [
            {"name": "web_search", "args": {"query": "vector databases"}, "id": "web-live-1", "type": "tool_call"},
            {"name": "arxiv_tool", "args": {"query": "vector databases"}, "id": "arxiv-live-1", "type": "tool_call"},
        ]
        observed = {}

        class FakeToolChatModel(BaseChatModel):
            @property
            def _llm_type(self):
                return "checkpoint-regression-test"

            def bind_tools(self, tools, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                observed["messages"] = list(messages)
                return ChatResult(generations=[ChatGeneration(
                    message=AIMessage(content="The checkpoint was repaired before the model ran.")
                )])

        @tool
        def web_search(query: str) -> str:
            """Return a test web result."""
            return "web result"

        @tool
        def arxiv_tool(query: str) -> str:
            """Return a test paper result."""
            return "paper result"

        graph = create_react_agent(
            model=FakeToolChatModel(),
            tools=[web_search, arxiv_tool],
            pre_model_hook=agent._close_incomplete_tool_calls,
            checkpointer=MemorySaver(),
        )
        config = {"configurable": {"thread_id": "partial-live-tool-batch"}}
        graph.update_state(
            config,
            {"messages": [
                HumanMessage(content="What are vector database trade-offs?"),
                AIMessage(content="", tool_calls=calls),
                ToolMessage(content="Web sources found.", tool_call_id="web-live-1", name="web_search"),
                ToolMessage(content="duplicate stale web result", tool_call_id="web-live-1", name="web_search"),
            ]},
            as_node="tools",
        )

        result = graph.invoke({"messages": []}, config=config)

        tool_messages = [m for m in observed["messages"] if isinstance(m, ToolMessage)]
        counts = {}
        for message in tool_messages:
            counts[message.tool_call_id] = counts.get(message.tool_call_id, 0) + 1
        self.assertEqual(counts, {"web-live-1": 1, "arxiv-live-1": 1})
        missing = next(message for message in tool_messages if message.tool_call_id == "arxiv-live-1")
        self.assertEqual(missing.status, "error")
        self.assertIn("did not finish", missing.content)
        self.assertEqual(result["messages"][-1].content, "The checkpoint was repaired before the model ran.")
        saved = graph.get_state(config).values["messages"]
        self.assertEqual(saved[0].content, "What are vector database trade-offs?")
        self.assertEqual(sum(isinstance(m, ToolMessage) and m.tool_call_id == "arxiv-live-1" for m in saved), 1)

    def test_checkpoint_repair_routes_saved_tool_call_back_to_model(self):
        calls = [
            {"name": "web_search", "args": {"query": "saved query"}, "id": "saved-1", "type": "tool_call"},
            {"name": "arxiv_tool", "args": {"query": "saved papers"}, "id": "saved-2", "type": "tool_call"},
        ]

        def model_node(state):
            answered = {
                message.tool_call_id
                for message in state["messages"]
                if isinstance(message, ToolMessage)
            }
            if not all(call["id"] in answered for call in calls):
                raise AssertionError("model was invoked before all saved tool calls had results")
            return {"messages": [AIMessage(content="Checkpoint recovered without losing history.")]}

        def route_after_model(state):
            last_message = state["messages"][-1]
            return "tools" if getattr(last_message, "tool_calls", []) else END

        builder = StateGraph(MessagesState)
        builder.add_node("agent", model_node)
        builder.add_node("tools", ToolNode([]))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", route_after_model, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        graph = builder.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "incomplete-checkpoint-test"}}
        graph.update_state(
            config,
            {
                "messages": [
                    HumanMessage(content="Earlier thread history"),
                    AIMessage(content="", tool_calls=calls),
                ]
            },
            as_node="agent",
        )

        with patch("agent.agent", graph):
            result = agent.run_agent("Continue the discussion", "incomplete-checkpoint-test")

        self.assertEqual(result["answer"], "Checkpoint recovered without losing history.")
        # Repaired calls belong to the earlier checkpoint turn; the API trace
        # for this request must not present old tool calls as current attempts.
        self.assertEqual(result["steps"], [])
        saved_messages = graph.get_state(config).values["messages"]
        self.assertEqual(saved_messages[0].content, "Earlier thread history")
        self.assertEqual(
            {message.tool_call_id for message in saved_messages if isinstance(message, ToolMessage)},
            {"saved-1", "saved-2"},
        )

    def test_run_agent_repairs_every_unanswered_call_and_resumes_without_clearing_history(self):
        calls = [
            {"name": "web_search", "args": {"query": "fresh sources"}, "id": "call-1", "type": "tool_call"},
            {"name": "arxiv_tool", "args": {"query": "research"}, "id": "call-2", "type": "tool_call"},
        ]

        class InterruptedGraph:
            def __init__(self):
                self.messages = [HumanMessage(content="Earlier turn")]
                self.update_as_node = None
                self.invoke_inputs = []

            def get_state(self, config):
                return SimpleNamespace(values={"messages": list(self.messages)})

            def update_state(self, config, values, *, as_node):
                self.update_as_node = as_node
                self.messages.extend(values["messages"])

            def invoke(self, values, config):
                self.invoke_inputs.append(values)
                if len(self.invoke_inputs) == 1:
                    self.messages.append(HumanMessage(content="Find fresh research"))
                    self.messages.append(AIMessage(content="", tool_calls=calls))
                    raise TimeoutError("simulated interruption after model requested tools")

                completed_ids = {
                    message.tool_call_id
                    for message in self.messages
                    if isinstance(message, ToolMessage)
                }
                if not all(call["id"] in completed_ids for call in calls):
                    raise AssertionError("model resumed with an incomplete tool-call batch")
                return {"messages": self.messages + [AIMessage(content="Recovered with tool errors recorded.")]}

        graph = InterruptedGraph()
        with patch("agent.agent", graph):
            result = agent.run_agent("Find fresh research", "test-thread")

        self.assertEqual(result["answer"], "Recovered with tool errors recorded.")
        self.assertEqual(len(result["steps"]), 2)
        self.assertEqual(graph.update_as_node, "tools")
        self.assertEqual(graph.invoke_inputs[1], {"messages": []})
        self.assertEqual(graph.messages[0].content, "Earlier turn")
        repaired = [message for message in graph.messages if isinstance(message, ToolMessage)]
        self.assertEqual({message.tool_call_id for message in repaired}, {"call-1", "call-2"})
        self.assertTrue(all(message.status == "error" for message in repaired))


if __name__ == "__main__":
    unittest.main()
