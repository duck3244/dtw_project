"""ASGI middleware that rejects oversized request bodies before the handler runs.

Two layers of defence:
  (1) Pre-check the Content-Length header (cheap, catches well-behaved clients).
  (2) Wrap `receive` so a body that streams past the cap also gets a 413 instead
      of being passed in full to FastAPI multipart parsing.

For real protection against DoS (chunked uploads with no length, etc.), pair
this with an upstream reverse-proxy limit, e.g. nginx `client_max_body_size`.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send


_OVER_LIMIT_BODY = b'{"detail":"audio too large"}'
_GUARDED_METHODS = {"POST", "PUT", "PATCH"}


async def _reply_413(send: Send) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_OVER_LIMIT_BODY)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": _OVER_LIMIT_BODY})


class BodySizeLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int,
        route_overrides: dict[str, int] | None = None,
    ) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)
        self.overrides = {p: int(b) for p, b in (route_overrides or {}).items()}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in _GUARDED_METHODS:
            await self.app(scope, receive, send)
            return

        cap = self.overrides.get(scope.get("path", ""), self.max_bytes)

        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > cap:
                        await _reply_413(send)
                        return
                except ValueError:
                    pass
                break

        seen = 0
        rejected = False
        replied = False

        async def guarded_receive() -> Message:
            nonlocal seen, rejected
            if rejected:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > cap:
                    rejected = True
                    return {"type": "http.disconnect"}
            return msg

        async def guarded_send(msg: Message) -> None:
            nonlocal replied
            if rejected and not replied:
                replied = True
                await _reply_413(send)
                return
            if replied:
                # downstream may keep emitting after the disconnect — drop
                return
            await send(msg)

        await self.app(scope, guarded_receive, guarded_send)
