import os
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

# -------------------- config --------------------

SERVERS_FILE = os.getenv("SERVERS_FILE", "servers.txt")

DEFAULT_PATH = os.getenv("DEFAULT_PATH", "/v1/completions")
DEFAULT_METHOD = os.getenv("DEFAULT_METHOD", "POST")
DEFAULT_TIMEOUT_S = float(os.getenv("DEFAULT_TIMEOUT_S", "600.0"))

# ВАЖНО: 10 параллельных задач на один vLLM сервер (модель)
MAX_INFLIGHT_PER_SERVER = int(os.getenv("MAX_INFLIGHT_PER_SERVER", "10"))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
UNHEALTHY_COOLDOWN_S = float(os.getenv("UNHEALTHY_COOLDOWN_S", "30.0"))

HTTPX_MAX_CONNECTIONS = int(os.getenv("HTTPX_MAX_CONNECTIONS", "2000"))
HTTPX_MAX_KEEPALIVE = int(os.getenv("HTTPX_MAX_KEEPALIVE", "200"))


# -------------------- models --------------------

@dataclass
class ServerState:
    url: str
    max_inflight: int = MAX_INFLIGHT_PER_SERVER
    inflight: int = 0
    unhealthy_until: float = 0.0
    last_error: str = ""
    rtt_ema: float = 0.0

    def available(self) -> bool:
        now = time.time()
        if now < self.unhealthy_until:
            return False
        return self.inflight < self.max_inflight

    def mark_bad(self, err: str):
        self.last_error = err
        self.unhealthy_until = time.time() + UNHEALTHY_COOLDOWN_S


@dataclass
class ConnectionState:
    ws: WebSocket
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class WSRequest(BaseModel):
    type: str = Field(default="request", examples=["request"])
    path: str = Field(default=DEFAULT_PATH, examples=["/v1/completions", "/v1/chat/completions"])
    method: str = Field(default=DEFAULT_METHOD, examples=["POST"])
    payload: Dict[str, Any] = Field(..., description="OpenAI-compatible JSON body for vLLM")
    headers: Optional[Dict[str, str]] = Field(default=None)
    timeout_s: float = Field(default=DEFAULT_TIMEOUT_S)


@dataclass
class Job:
    session_id: str
    conn: ConnectionState
    req: WSRequest
    created_at: float = field(default_factory=time.time)


# -------------------- app state --------------------

app = FastAPI(title="vLLM Smart Scheduler (WebSocket)", version="1.1")

servers: List[ServerState] = []
servers_lock = asyncio.Lock()
server_available = asyncio.Condition()

pending: asyncio.Queue[Job] = asyncio.Queue()

results: Dict[str, Dict[str, Any]] = {}
results_lock = asyncio.Lock()


# -------------------- utils --------------------

def load_servers(file_path: str) -> List[ServerState]:
    """
    servers.txt:
      http://10.0.0.1:8000
      http://10.0.0.1:8001
      ...
    """
    out: List[ServerState] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(ServerState(url=s.rstrip("/"), max_inflight=MAX_INFLIGHT_PER_SERVER))
    if not out:
        raise RuntimeError(f"No servers found in {file_path}")
    return out


async def ws_send(conn: ConnectionState, msg: Dict[str, Any]) -> bool:
    """
    Безопасная отправка в WS.
    Если клиент отвалился/соединение закрыто — не валим scheduler и не оставляем inflight зависшим.
    """
    try:
        async with conn.send_lock:
            await conn.ws.send_json(msg)
        return True
    except Exception:
        return False


async def pick_server() -> Optional[ServerState]:
    async with servers_lock:
        avail = [s for s in servers if s.available()]
        if not avail:
            return None

        def key(s: ServerState):
            rtt = s.rtt_ema if s.rtt_ema > 0 else 1e9
            return (s.inflight, rtt)

        chosen = min(avail, key=key)
        chosen.inflight += 1
        return chosen


async def release_server(s: ServerState):
    async with servers_lock:
        s.inflight = max(0, s.inflight - 1)
    async with server_available:
        server_available.notify_all()


async def forward_to_vllm(
    client: httpx.AsyncClient,
    server: ServerState,
    req: WSRequest,
) -> Dict[str, Any]:
    url = f"{server.url}{req.path}"
    t0 = time.time()
    try:
        resp = await client.request(
            method=req.method.upper(),
            url=url,
            json=req.payload,
            headers=req.headers,
            timeout=req.timeout_s,
        )

        # считаем 5xx как проблемы сервера (временно unhealthy)
        if resp.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"{server.url} returned {resp.status_code}",
                request=resp.request,
                response=resp,
            )

        data = resp.json()

        rtt = time.time() - t0
        alpha = 0.2
        server.rtt_ema = rtt if server.rtt_ema == 0 else (alpha * rtt + (1 - alpha) * server.rtt_ema)
        return data

    except Exception as e:
        server.mark_bad(str(e))
        raise


async def process_job(job: Job, server: ServerState, client: httpx.AsyncClient):
    """
    Обработка одного job.
    ВАЖНО: release_server() должен сработать ВСЕГДА, даже если клиент WS пропал.
    """
    started = time.time()
    current_server: ServerState = server

    try:
        await ws_send(job.conn, {
            "type": "accepted",
            "session_id": job.session_id,
            "server": current_server.url,
            "message": "accepted_in_work",
            "can_send_new": True,
        })

        last_err: Optional[str] = None
        result: Optional[Dict[str, Any]] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                result = await forward_to_vllm(client, current_server, job.req)
                last_err = None
                break
            except Exception as e:
                last_err = str(e)

                if attempt >= MAX_RETRIES:
                    break

                # освободить текущий сервер и попытаться взять другой
                await release_server(current_server)

                while True:
                    new_server = await pick_server()
                    if new_server is not None:
                        current_server = new_server
                        await ws_send(job.conn, {
                            "type": "rerouted",
                            "session_id": job.session_id,
                            "server": current_server.url,
                            "message": "rerouted_to_another_server",
                        })
                        break
                    async with server_available:
                        await server_available.wait()

        finished = time.time()

        if last_err is not None or result is None:
            await ws_send(job.conn, {
                "type": "error",
                "session_id": job.session_id,
                "error": last_err or "unknown_error",
                "server": current_server.url,
                "started_at": started,
                "finished_at": finished,
            })
            async with results_lock:
                results[job.session_id] = {
                    "status": "error",
                    "error": last_err or "unknown_error",
                    "server": current_server.url,
                    "started_at": started,
                    "finished_at": finished,
                }
            return

        await ws_send(job.conn, {
            "type": "done",
            "session_id": job.session_id,
            "server": current_server.url,
            "result": result,
            "started_at": started,
            "finished_at": finished,
        })

        async with results_lock:
            results[job.session_id] = {
                "status": "done",
                "server": current_server.url,
                "result": result,
                "started_at": started,
                "finished_at": finished,
            }

    finally:
        # если мы уже release делали при reroute — тут второй release не сломает (max(0, ...))
        await release_server(current_server)


async def dispatcher_loop():
    limits = httpx.Limits(max_connections=HTTPX_MAX_CONNECTIONS, max_keepalive_connections=HTTPX_MAX_KEEPALIVE)
    async with httpx.AsyncClient(limits=limits) as client:
        while True:
            job = await pending.get()
            try:
                while True:
                    server = await pick_server()
                    if server is not None:
                        asyncio.create_task(process_job(job, server, client))
                        break
                    async with server_available:
                        await server_available.wait()
            finally:
                pending.task_done()


# -------------------- endpoints --------------------

@app.on_event("startup")
async def startup_event():
    global servers
    servers = load_servers(SERVERS_FILE)
    asyncio.create_task(dispatcher_loop())


@app.get("/servers")
async def get_servers():
    async with servers_lock:
        return [
            {
                "url": s.url,
                "inflight": s.inflight,
                "max_inflight": s.max_inflight,          # теперь будет 10
                "unhealthy_until": s.unhealthy_until,
                "last_error": s.last_error,
                "rtt_ema": s.rtt_ema,
                "available_slots": max(0, s.max_inflight - s.inflight),
            }
            for s in servers
        ]


@app.post("/reload_servers")
async def reload_servers():
    global servers
    new_list = load_servers(SERVERS_FILE)
    async with servers_lock:
        servers = new_list
    async with server_available:
        server_available.notify_all()
    return {"ok": True, "count": len(servers)}


@app.get("/result/{session_id}")
async def get_result(session_id: str):
    async with results_lock:
        if session_id not in results:
            raise HTTPException(status_code=404, detail="unknown session_id")
        return results[session_id]


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    conn = ConnectionState(ws=ws)

    try:
        while True:
            msg = await ws.receive_json()
            try:
                req = WSRequest(**msg)
            except Exception as e:
                await ws_send(conn, {"type": "error", "error": f"bad_request: {e}"})
                continue

            if req.type != "request":
                await ws_send(conn, {"type": "error", "error": "unsupported_message_type"})
                continue

            session_id = str(uuid.uuid4())
            job = Job(session_id=session_id, conn=conn, req=req)

            async with servers_lock:
                any_free = any(s.available() for s in servers)

            # ack клиенту (как у тебя было)
            if not any_free:
                await ws_send(conn, {
                    "type": "busy",
                    "session_id": session_id,
                    "message": "all_servers_busy_wait",
                    "can_send_new": False,
                })
            else:
                await ws_send(conn, {
                    "type": "received",
                    "session_id": session_id,
                    "message": "received_ok",
                })

            await pending.put(job)

    except WebSocketDisconnect:
        return