from dataclasses import dataclass
import hmac
import os
import threading

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class Principal:
    subject: str
    instance_admin: bool = False


PERMISSIONS = {
    "viewer": {"read"},
    "researcher": {"read", "train", "intervene", "export"},
    "admin": {"read", "train", "intervene", "export", "manage"},
}


class Authenticator:
    def __init__(self):
        self.token = os.getenv("NEURALIS_TOKEN", "")
        self.issuer = os.getenv("NEURALIS_OIDC_ISSUER", "")
        self.audience = os.getenv("NEURALIS_OIDC_AUDIENCE", "")
        self.jwks = os.getenv("NEURALIS_OIDC_JWKS_URL", "")
        self._jwks_client = None
        self._lock = threading.Lock()
        if any([self.issuer, self.audience, self.jwks]) and not all([self.issuer, self.audience, self.jwks]):
            raise ValueError("OIDC requires issuer, audience, and HTTPS JWKS URL together")
        if self.jwks and not self.jwks.startswith("https://"):
            raise ValueError("OIDC JWKS URL must use HTTPS")
        host = os.getenv("NEURALIS_HOST", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost", "::1"} and not (self.token or self.issuer):
            raise ValueError("A non-loopback backend requires NEURALIS_TOKEN or configured OIDC")

    def authenticate(self, authorization: str | None) -> Principal:
        if not self.token and not self.issuer:
            return Principal("local", True)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Bearer authentication required", headers={"WWW-Authenticate": "Bearer"})
        credential = authorization.removeprefix("Bearer ")
        if self.token and hmac.compare_digest(credential, self.token):
            return Principal("instance-token", True)
        if not self.issuer:
            raise HTTPException(401, "Invalid bearer credential")
        try:
            import jwt
            with self._lock:
                if self._jwks_client is None:
                    self._jwks_client = jwt.PyJWKClient(self.jwks, cache_keys=True, lifespan=300, timeout=5)
            key = self._jwks_client.get_signing_key_from_jwt(credential)
            payload = jwt.decode(credential, key.key, algorithms=["RS256", "ES256"],
                                 issuer=self.issuer, audience=self.audience,
                                 options={"require": ["exp", "iss", "aud", "sub"]}, leeway=15)
            if not isinstance(payload["sub"], str) or not payload["sub"]:
                raise ValueError("Invalid OIDC subject")
            return Principal(payload["sub"], False)
        except ImportError as error:
            raise HTTPException(503, "OIDC dependency missing: install the lab extra") from error
        except Exception as error:
            raise HTTPException(401, "OIDC validation failed") from error

    def authorize(self, repository, principal, project_id, permission):
        if principal.instance_admin:
            return
        with repository.transaction() as connection:
            row = connection.execute("SELECT role FROM members WHERE project_id=? AND subject=?",
                                     (project_id, principal.subject)).fetchone()
        if row is None or permission not in PERMISSIONS.get(row["role"], set()):
            repository.audit(principal.subject, project_id, permission, project_id, "denied")
            raise HTTPException(403, "Project permission denied")


def principal_for(request: Request):
    return request.app.state.auth.authenticate(request.headers.get("authorization"))
