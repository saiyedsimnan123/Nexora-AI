"""HTTP security headers middleware and response protections for Nexora AI."""

from __future__ import annotations

import os
from typing import Any, Mapping

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Content security policy tailored to React 19 + Three.js 3D canvas and same-origin API
DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self' data:; "
    "connect-src 'self' https:; "
    "frame-ancestors 'none'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self';"
)

DEFAULT_PERMISSIONS_POLICY = (
    "accelerometer=(), "
    "camera=(), "
    "geolocation=(), "
    "gyroscope=(), "
    "magnetometer=(), "
    "microphone=(), "
    "payment=(), "
    "usb=()"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces defense-in-depth HTTP security headers across all responses."""

    def __init__(
        self,
        app: Any,
        *,
        csp: str | None = DEFAULT_CSP,
        enable_hsts: bool | None = None,
        hsts_max_age: int = 31536000,
        include_subdomains: bool = True,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(app)
        self.csp = csp
        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age
        self.include_subdomains = include_subdomains
        self.environ = os.environ if environ is None else environ

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)

        # 1. MIME-type sniffing protection
        response.headers.setdefault("X-Content-Type-Options", "nosniff")

        # 2. Clickjacking protection (block embedding)
        response.headers.setdefault("X-Frame-Options", "DENY")

        # 3. Referrer privacy protection
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")

        # 4. Restrict dangerous browser features and sensors
        response.headers.setdefault("Permissions-Policy", DEFAULT_PERMISSIONS_POLICY)

        # 5. Content Security Policy
        if self.csp:
            response.headers.setdefault("Content-Security-Policy", self.csp)

        # 6. Strict-Transport-Security (HSTS)
        # Guard: Only emit HSTS over genuine HTTPS connections or when explicitly configured,
        # never for insecure local HTTP development.
        is_https = (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto", "").lower() == "https"
        )
        should_emit_hsts = (
            self.enable_hsts is True
            or (self.enable_hsts is None and is_https)
        )
        if should_emit_hsts and is_https:
            hsts_val = f"max-age={self.hsts_max_age}"
            if self.include_subdomains:
                hsts_val += "; includeSubDomains"
            response.headers.setdefault("Strict-Transport-Security", hsts_val)

        return response
