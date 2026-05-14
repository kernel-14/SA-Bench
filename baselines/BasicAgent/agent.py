from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from baselines.BasicAgent.models import AgentResult
from baselines.BasicAgent.prompts import DEFAULT_CONTINUE_MESSAGE
from baselines.BasicAgent.recording import append_jsonl, append_text, utc_now_text
from baselines.BasicAgent.tools import BaseTool, ToolResult


def should_stop(
    num_steps: int,
    max_steps: int | None,
    start_time: float,
    time_limit: int | None,
) -> bool:
    if max_steps is not None and num_steps > max_steps:
        return True
    if time_limit is not None and (time.time() - start_time) > time_limit:
        return True
    return False


def _assistant_to_dict(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", None) or ""
    if not content and not getattr(message, "tool_calls", None):
        content = "OK."  # Gemini requires at least one non-empty part
    payload: dict[str, Any] = {"role": message.role, "content": content}
    if getattr(message, "tool_calls", None):
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _prune_messages(messages: list[dict[str, Any]], max_messages: int = 200) -> list[dict[str, Any]]:
    if len(messages) <= max_messages:
        return messages
    system_messages = [message for message in messages if message["role"] == "system"]
    non_system_messages = [message for message in messages if message["role"] != "system"]

    first_user_message: dict[str, Any] | None = next(
        (message for message in non_system_messages if message["role"] == "user"),
        None,
    )

    budget = max(max_messages - len(system_messages), 1)
    trimmed_non_system = non_system_messages[-budget:]
    if first_user_message is not None and first_user_message not in trimmed_non_system:
        trimmed_non_system = [first_user_message] + trimmed_non_system[:-1]

    valid_messages: list[dict[str, Any]] = []
    active_tool_ids: set[str] = set()

    for message in trimmed_non_system:
        role = message["role"]
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            active_tool_ids = {tool_call["id"] for tool_call in tool_calls}
            valid_messages.append(message)
            continue
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id in active_tool_ids:
                valid_messages.append(message)
            continue
        valid_messages.append(message)
        active_tool_ids = set()

    # Ensure no trailing assistant message has dangling tool_calls
    # (tool messages after the budget boundary may have been trimmed)
    while valid_messages and valid_messages[-1].get("role") == "assistant":
        tool_calls = valid_messages[-1].get("tool_calls") or []
        if not tool_calls:
            break
        expected_ids = {tc["id"] for tc in tool_calls}
        found_ids: set[str] = set()
        for msg in reversed(valid_messages):
            if msg.get("role") == "tool":
                found_ids.add(msg.get("tool_call_id", ""))
            elif msg.get("role") == "assistant" and msg is not valid_messages[-1]:
                break
        if expected_ids - found_ids:
            valid_messages.pop()
        else:
            break

    return system_messages + valid_messages


@dataclass
class BasicAgent:
    client: OpenAI
    model_name: str
    tools: list[BaseTool]
    max_steps: int
    time_limit_seconds: int
    messages_path: Path
    agent_log_path: Path

    def run(self, workspace_root: Path, prompt_messages: list[dict[str, Any]]) -> AgentResult:
        messages = list(prompt_messages)
        tool_map = {tool.name(): tool for tool in self.tools}
        tool_trace: list[dict[str, Any]] = []
        start_time = time.time()
        steps = 0
        submitted = False
        submission_note = ""
        total_input_tokens = 0
        total_output_tokens = 0
        step_timings: list[dict[str, Any]] = []

        while not should_stop(steps, self.max_steps, start_time, self.time_limit_seconds):
            steps += 1
            step_start = time.time()
            messages = _prune_messages(messages)
            kwargs: dict[str, Any] = {}
            if "deepseek" in self.model_name.lower():
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=[tool.schema() for tool in self.tools],
                tool_choice="auto",
                **kwargs,
            )
            step_elapsed = time.time() - step_start

            usage = response.usage
            step_input = getattr(usage, "prompt_tokens", 0) or 0
            step_output = getattr(usage, "completion_tokens", 0) or 0
            total_input_tokens += step_input
            total_output_tokens += step_output
            step_timings.append({
                "step": steps,
                "elapsed_s": round(step_elapsed, 2),
                "input_tokens": step_input,
                "output_tokens": step_output,
            })

            message = response.choices[0].message
            assistant_payload = _assistant_to_dict(message)
            append_jsonl(self.messages_path, {"time": utc_now_text(), "message": assistant_payload})
            append_text(
                self.agent_log_path,
                f"\n## Step {steps} ({step_elapsed:.1f}s, in={step_input} out={step_output})\n\n"
                f"[assistant]\n{json.dumps(assistant_payload, ensure_ascii=False, indent=2)}\n",
            )
            messages.append(assistant_payload)

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                follow_up = {"role": "user", "content": DEFAULT_CONTINUE_MESSAGE}
                messages.append(follow_up)
                append_jsonl(self.messages_path, {"time": utc_now_text(), "message": follow_up})
                continue

            for tool_call in tool_calls:
                tool = tool_map[tool_call.function.name]
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_payload = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": (
                            "JSONDecodeError: invalid JSON in tool call arguments. "
                            "Check for unescaped newlines, quotes, or control characters. "
                            "Re-encode the command/arguments as proper JSON."
                        ),
                    }
                    messages.append(tool_payload)
                    append_jsonl(self.messages_path, {"time": utc_now_text(), "message": tool_payload})
                    tool_trace.append({
                        "step": steps,
                        "tool_name": tool.name(),
                        "arguments": {"error": "json_decode_error"},
                        "result_preview": "JSON parse error",
                    })
                    continue
                try:
                    result = tool.execute(workspace_root=workspace_root, **arguments)
                except Exception as tool_exc:
                    result = ToolResult(content=f"ToolError: {tool_exc}")
                tool_payload = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content,
                }
                messages.append(tool_payload)
                append_jsonl(self.messages_path, {"time": utc_now_text(), "message": tool_payload})
                tool_trace.append(
                    {
                        "step": steps,
                        "tool_name": tool.name(),
                        "arguments": arguments,
                        "result_preview": result.content[:500],
                    }
                )
                if result.submitted:
                    submitted = True
                    submission_note = result.submission_note
                    return AgentResult(
                        submitted=submitted,
                        submission_note=submission_note,
                        steps=steps,
                        messages=messages,
                        tool_trace=tool_trace,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                        elapsed_seconds=round(time.time() - start_time, 1),
                        step_timings=step_timings,
                    )

        return AgentResult(
            submitted=submitted,
            submission_note=submission_note,
            steps=steps,
            messages=messages,
            tool_trace=tool_trace,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            elapsed_seconds=round(time.time() - start_time, 1),
            step_timings=step_timings,
        )
