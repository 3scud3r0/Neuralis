from starlette.responses import JSONResponse


class BodyLimitMiddleware:
    """Enforce the import budget even when the sender uses chunked transfer."""

    def __init__(self, app, maximum_bytes=8 * 1024 * 1024):
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PATCH", "PUT"}:
            await self.app(scope, receive, send)
            return
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.maximum_bytes:
                response = JSONResponse({"category": "request-too-large", "message": "Request exceeds the eight-MiB import budget"}, status_code=413)
                await response(scope, receive, send)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break
        index = 0
        async def replay():
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return await receive()
        await self.app(scope, replay, send)
