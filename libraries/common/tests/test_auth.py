"""
Tests that access tokens are validated end to end through authutils.

Nothing here mocks authutils: tokens are signed with a throwaway RSA key and served from a local
issuer, so the OIDC discovery, JWKS fetch, signature check and claim checks all run for real.
That is what catches an authutils upgrade that changes how a token is accepted or rejected.
"""

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jwt.algorithms import RSAAlgorithm
from starlette.status import HTTP_401_UNAUTHORIZED

from common import auth, config

KEY_ID = "test-key"
SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
# Not served by the issuer, so a token it signs fails signature verification.
UNKNOWN_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def issuer() -> Iterator[str]:
    """
    Serve OIDC discovery and a JWKS for SIGNING_KEY from a local issuer.

    Bound to a free port, so each run gets an issuer URL nothing else has used. authutils caches
    public keys per issuer for the life of the process, and a fresh URL keeps that cache from
    carrying over between runs in the same worker.

    Yields:
        str: The issuer URL, as it appears in a token's `iss` claim.
    """
    jwk = json.loads(RSAAlgorithm.to_jwk(SIGNING_KEY.public_key()))
    jwk.update(kid=KEY_ID, alg="RS256", use="sig")
    server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    issuer_url = f"http://127.0.0.1:{server.server_port}/user"

    class IssuerHandler(BaseHTTPRequestHandler):
        """Answers the two requests authutils makes to find an issuer's keys."""

        def do_GET(self) -> None:  # noqa: N802 - name is fixed by BaseHTTPRequestHandler
            """Return the discovery document or the key set, depending on the path."""
            if self.path.endswith("/.well-known/openid-configuration"):
                body = {"issuer": issuer_url, "jwks_uri": f"{issuer_url}/jwt/keys"}
            else:
                body = {"keys": [jwk]}
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - overridden signature
            """Keep request lines out of the test output."""

    server.RequestHandlerClass = IssuerHandler
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield issuer_url
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def enforce_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every test validate its token, whatever the environment sets."""
    monkeypatch.setattr(config, "DEBUG_SKIP_AUTH", False)
    monkeypatch.setattr(config, "ALLOW_ANONYMOUS_ACCESS", False)


def make_token(issuer: str, key: rsa.RSAPrivateKey = SIGNING_KEY, **claims: Any) -> HTTPAuthorizationCredentials:
    """
    Build a bearer token shaped like a Gen3 access token.

    Args:
        issuer (str): Value for the `iss` claim.
        key (rsa.RSAPrivateKey): Key to sign with.
        **claims (Any): Claims to add or override.

    Returns:
        HTTPAuthorizationCredentials: The token as FastAPI's bearer dependency would hand it over.
    """
    now = int(time.time())
    payload = {
        "iss": issuer,
        "sub": "42",
        "aud": ["gen3"],
        "scope": ["user", "openid"],
        "pur": "access",
        "iat": now,
        "exp": now + 300,
        "jti": "test-jti",
        "context": {"user": {"name": "test-user"}},
    }
    payload.update(claims)
    encoded = jwt.encode(payload, key, algorithm="RS256", headers={"kid": KEY_ID})
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=encoded)


@pytest.mark.asyncio
async def test_valid_token_yields_user_id(issuer: str) -> None:
    """A valid access token resolves to its `sub` claim."""
    assert await auth.get_user_id(make_token(issuer)) == "42"


@pytest.mark.asyncio
async def test_valid_token_yields_username(issuer: str) -> None:
    """A valid access token resolves to the user name in its context claim."""
    assert await auth.get_username(make_token(issuer)) == "test-user"


@pytest.mark.parametrize(
    ("case", "overrides"),
    [
        ("wrong purpose", {"pur": "refresh"}),
        ("missing scope", {"scope": ["user"]}),
        ("wrong audience", {"aud": ["not-gen3"]}),
        ("expired", {"iat": int(time.time()) - 600, "exp": int(time.time()) - 300}),
        ("unknown signing key", {"key": UNKNOWN_KEY}),
    ],
)
@pytest.mark.asyncio
async def test_invalid_token_is_rejected(issuer: str, case: str, overrides: dict[str, Any]) -> None:
    """A token failing any one validation check is answered with a 401."""
    token = make_token(issuer, **overrides)

    with pytest.raises(HTTPException) as exc_info:
        await auth.get_user_id(token)

    assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED, case


@pytest.mark.asyncio
async def test_missing_token_is_rejected() -> None:
    """With no token and no request to read one from, the caller is unauthenticated."""
    with pytest.raises(HTTPException) as exc_info:
        await auth.get_user_id()

    assert exc_info.value.status_code == HTTP_401_UNAUTHORIZED
