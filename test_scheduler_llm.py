from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets


DEFAULT_WS_URL = "ws://10.233.3.148:9000/ws"
DEFAULT_MODEL = "Llama-3_3-Nemotron-Super-49B-v1_5"
DEFAULT_PROMPT = (
    "Ты тестовый LLM endpoint. Ответь строго одним JSON без markdown и без пояснений: "
    '{"ok":1,"answer":"pong"}'
)


@dataclass(slots=True)
class SchedulerTaskHandle:
    session_id: str
    done_future: Any

    async def result(self) -> dict[str, Any]:
        return await self.done_future


class SchedulerClient:
    def __init__(
        self,
        ws_url: str,
        model: str,
        path: str = "/v1/completions",
        method: str = "POST",
        open_timeout_seconds: float = 30.0,
    ) -> None:
        self.ws_url = ws_url
        self.model = model
        self.path = path
        self.method = method
        self.open_timeout_seconds = open_timeout_seconds
        self.ws = None
        self.recv_task = None
        self.submit_lock = asyncio.Lock()
        self.send_lock = asyncio.Lock()
        self.awaiting_ack = None
        self.done_futures: dict[str, Any] = {}
        self.early_msgs: dict[str, list[dict[str, Any]]] = {}

    async def connect(self) -> None:
        self.ws = await websockets.connect(
            self.ws_url,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
            open_timeout=float(self.open_timeout_seconds),
        )
        self.recv_task = asyncio.create_task(self.receiver_loop())

    async def close(self) -> None:
        if self.recv_task:
            self.recv_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self.recv_task
        if self.ws:
            with contextlib.suppress(Exception):
                await self.ws.close()

    def fail_pending(self, error: BaseException) -> None:
        if self.awaiting_ack and not self.awaiting_ack.done():
            self.awaiting_ack.set_exception(error)
        for future in list(self.done_futures.values()):
            if not future.done():
                future.set_exception(error)

    async def ws_send(self, message: dict[str, Any]) -> None:
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        async with self.send_lock:
            await self.ws.send(json.dumps(message, ensure_ascii=False))

    def handle_session_msg(self, session_id: str, message: dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type in {"accepted", "rerouted"}:
            return
        if msg_type == "done":
            future = self.done_futures.get(session_id)
            if future and not future.done():
                future.set_result(message)
            return
        if msg_type == "error":
            future = self.done_futures.get(session_id)
            if future and not future.done():
                future.set_exception(RuntimeError(str(message.get("error") or "job_failed")))

    async def receiver_loop(self) -> None:
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        try:
            async for raw_message in self.ws:
                message = json.loads(raw_message)
                msg_type = message.get("type")
                if msg_type in {"received", "busy"}:
                    if self.awaiting_ack and not self.awaiting_ack.done():
                        self.awaiting_ack.set_result(message)
                    continue
                session_id = message.get("session_id")
                if not session_id:
                    continue
                if msg_type in {"accepted", "rerouted", "done", "error"}:
                    self.handle_session_msg(session_id, message)
                else:
                    self.early_msgs.setdefault(session_id, []).append(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail_pending(RuntimeError(f"Scheduler WebSocket receiver failed: {exc}"))

    async def submit(
        self,
        *,
        prompt: str,
        text: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> SchedulerTaskHandle:
        if not self.ws or (self.recv_task is not None and self.recv_task.done()):
            raise RuntimeError("WebSocket is not connected")
        payload = {
            "model": self.model,
            "prompt": f"{prompt}\n\n{text}\n",
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        request = {
            "type": "request",
            "path": self.path,
            "method": self.method,
            "payload": payload,
            "timeout_s": float(timeout_s),
        }
        async with self.submit_lock:
            self.awaiting_ack = asyncio.get_running_loop().create_future()
            await self.ws_send(request)
            ack = await self.awaiting_ack
            session_id = str(ack["session_id"])
            done_future = self.done_futures.setdefault(session_id, asyncio.get_running_loop().create_future())
            for early_msg in self.early_msgs.pop(session_id, []):
                self.handle_session_msg(session_id, early_msg)
            return SchedulerTaskHandle(session_id=session_id, done_future=done_future)


def extract_completion_text(result: dict[str, Any]) -> str:
    payload = result.get("result", result)
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices or not isinstance(choices, list):
        return ""
    first_choice = choices[0] if choices else {}
    if not isinstance(first_choice, dict):
        return ""
    text = first_choice.get("text", "")
    return text if isinstance(text, str) else ""


def build_text(request_index: int, args: argparse.Namespace) -> str:
    if args.text_file:
        template = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text is not None:
        template = args.text
    else:
        template = (
            f"Тестовый запрос #{request_index}. "
            'Верни ровно JSON {"ok":1,"request_index":'
            f"{request_index}"
            ',"answer":"pong"}'
        )
    return template.replace("{request_index}", str(request_index))


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


async def run_one(
    *,
    client: SchedulerClient,
    request_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    record: dict[str, Any] = {
        "request_index": request_index,
        "ok": False,
        "empty_text": True,
        "session_id": None,
        "server": None,
        "latency_seconds": None,
        "error": None,
        "completion_text": "",
        "raw_done_message": None,
    }
    prompt = resolve_prompt(args)
    text = build_text(request_index, args)
    record["prompt_chars"] = len(prompt)
    record["text_chars"] = len(text)
    record["full_prompt_chars"] = len(prompt) + 2 + len(text) + 1
    try:
        handle = await asyncio.wait_for(
            client.submit(
                prompt=prompt,
                text=text,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.scheduler_timeout_s,
            ),
            timeout=args.request_timeout_seconds,
        )
        record["session_id"] = handle.session_id
        done_message = await asyncio.wait_for(handle.result(), timeout=args.request_timeout_seconds)
        completion_text = extract_completion_text(done_message)
        record["ok"] = True
        record["empty_text"] = len(completion_text.strip()) == 0
        record["server"] = done_message.get("server")
        record["completion_text"] = completion_text
        record["raw_done_message"] = done_message
    except Exception as exc:
        record["error"] = str(exc)
    finally:
        record["latency_seconds"] = round(time.perf_counter() - started_at, 3)
    return record


async def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = output_dir / "scheduler_llm_smoke_raw.jsonl"

    client = SchedulerClient(
        args.ws_url,
        model=args.model,
        path=args.path,
        method=args.method,
        open_timeout_seconds=args.open_timeout_seconds,
    )
    try:
        await client.connect()
    except Exception as exc:
        error_record = {
            "ok": False,
            "stage": "connect",
            "ws_url": args.ws_url,
            "model": args.model,
            "error": str(exc),
        }
        with raw_log_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        print(f"Scheduler connection failed: {exc}", flush=True)
        print(f"Raw log: {raw_log_path.resolve()}", flush=True)
        return 1

    semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))

    async def limited_run(request_index: int) -> dict[str, Any]:
        async with semaphore:
            return await run_one(client=client, request_index=request_index, args=args)

    try:
        tasks = [asyncio.create_task(limited_run(index)) for index in range(args.requests)]
        results: list[dict[str, Any]] = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            status = "EMPTY" if result["empty_text"] and result["ok"] else "OK" if result["ok"] else "ERR"
            print(
                f"[{status}] request={result['request_index']} "
                f"session={result['session_id']} server={result['server']} "
                f"full_prompt_chars={result.get('full_prompt_chars')} "
                f"latency={result['latency_seconds']}s error={result['error']}",
                flush=True,
            )
    finally:
        await client.close()

    with raw_log_path.open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: int(item["request_index"])):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    ok_count = sum(1 for item in results if item["ok"])
    error_count = sum(1 for item in results if not item["ok"])
    empty_count = sum(1 for item in results if item["ok"] and item["empty_text"])
    non_empty_count = ok_count - empty_count
    servers = sorted({str(item["server"]) for item in results if item.get("server")})

    print("\nSummary:", flush=True)
    print(f"- requests: {len(results)}", flush=True)
    print(f"- ok: {ok_count}", flush=True)
    print(f"- non_empty_text: {non_empty_count}", flush=True)
    print(f"- empty_text: {empty_count}", flush=True)
    print(f"- errors: {error_count}", flush=True)
    print(f"- servers_seen: {', '.join(servers) if servers else 'none'}", flush=True)
    print(f"- raw_log: {raw_log_path.resolve()}", flush=True)

    return 1 if error_count or empty_count else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test scheduler WebSocket and LLM completion responses.")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="Scheduler WebSocket URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name passed to the scheduler payload.")
    parser.add_argument("--path", default="/v1/completions", help="OpenAI-compatible backend path.")
    parser.add_argument("--method", default="POST", help="HTTP method passed to scheduler.")
    parser.add_argument("--requests", type=int, default=8, help="How many test requests to send.")
    parser.add_argument("--concurrency", type=int, default=8, help="How many requests to keep in flight.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=128, help="max_tokens for each completion.")
    parser.add_argument("--scheduler-timeout-s", type=float, default=600.0, help="timeout_s forwarded to scheduler backend request.")
    parser.add_argument("--open-timeout-seconds", type=float, default=30.0, help="WebSocket opening handshake timeout.")
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0, help="Client-side wait timeout per submit/result.")
    parser.add_argument("--output-dir", default="scheduler_llm_smoke_test", help="Directory for raw JSONL logs.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="System-style prompt prepended before the test text.")
    parser.add_argument("--prompt-file", default=None, help="Read scheduler prompt from a UTF-8 text file.")
    parser.add_argument("--text", default=None, help="Request body text. {request_index} is replaced per request.")
    parser.add_argument("--text-file", default=None, help="Read request body text from a UTF-8 file. {request_index} is replaced per request.")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
