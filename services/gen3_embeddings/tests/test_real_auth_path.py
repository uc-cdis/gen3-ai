"""
Tests that exercise the real authorization check, not the test suite's bypass.

Every other test in this suite runs with `DEBUG_SKIP_AUTH` on and `get_allowed_authz_for_request`
faked, so `common.auth.authorize_request` returns before it does anything. That is deliberate --
it keeps the other tests about the behavior they are named for -- but it leaves the gate itself
unexercised, and a gate nothing tests is a gate that can silently stop closing.

Getting in means turning the bypass off: `common.auth` reads `common.config.DEBUG_SKIP_AUTH` as
an attribute at call time, so patching it on that module object reaches the running check.
Patching `gen3_embeddings.config` does NOT, which is exactly the gap that let CI answer 401 to
everything while the suite passed locally.

Sending a bearer token does not get you there, even though `common.config` says `DEBUG_SKIP_AUTH`
only applies "when a token is not provided". The bypass tests the `token` *argument*, which the
service never passes, and the request's `Authorization` header is not read until the line after
it -- so with the flag on, a real token is ignored rather than enforced. See
`test_documents_that_a_token_does_not_defeat_the_bypass`, which pins that as it currently stands.

The policy engine is faked at `app.state.arborist_client`, the same place the service reads it
from, so these tests assert against what the service *asked* the policy engine rather than
needing one running.
"""

import pytest

from common import auth as common_auth
from common import config as common_config
from gen3_embeddings.config import AUTHZ_SERVICE_NAME


class FakeArboristClient:
    """Records what it was asked to authorize and answers with a fixed verdict."""

    def __init__(self, verdict: bool):
        self.verdict = verdict
        self.calls: list[dict] = []

    async def auth_request(self, token, service, methods, resources):
        self.calls.append({"token": token, "service": service, "methods": methods, "resources": resources})
        return self.verdict


@pytest.fixture
def real_auth(app, monkeypatch):
    """
    Turn the suite's bypass off and install a fake policy engine with a fixed verdict.

    `get_user_id` only feeds a log line in `authorize_request`, and its real implementation
    reaches out to validate the token's claims. Stubbing it keeps these tests off the network
    without weakening what they check: the verdict comes from `auth_request`, which each test
    fakes explicitly.
    """

    def _install(verdict: bool) -> FakeArboristClient:
        async def fake_get_user_id(token=None, request=None, authz_config=None):
            return "test-user"

        monkeypatch.setattr(common_config, "DEBUG_SKIP_AUTH", False)
        monkeypatch.setattr(common_auth, "get_user_id", fake_get_user_id)
        arborist = FakeArboristClient(verdict)
        app.state.arborist_client = arborist
        return arborist

    return _install


def test_missing_token_is_rejected_when_the_bypass_is_off(client, allow_authz, monkeypatch):
    """
    With `DEBUG_SKIP_AUTH` off and no token, a collection route answers 401.

    This is the check the rest of the suite turns off. `allow_authz` is still applied so that the
    grant lookup succeeds and the 401 can only be coming from `authorize_request` itself.
    """
    monkeypatch.setattr(common_config, "DEBUG_SKIP_AUTH", False)
    allow_authz("docs")

    response = client.get("/vectorstore/collections/docs")

    assert response.status_code == 401, response.text


def test_a_denied_caller_gets_403(client, allow_authz, real_auth):
    """A caller the policy engine refuses is rejected, and the engine really was consulted."""
    allow_authz("docs")
    arborist = real_auth(verdict=False)

    response = client.get("/vectorstore/collections/docs", headers={"Authorization": "Bearer some-token"})

    assert response.status_code == 403, response.text
    assert len(arborist.calls) == 1, "the policy engine was not consulted"


def test_documents_that_a_token_does_not_defeat_the_bypass(client, allow_authz, app, monkeypatch):
    """
    With `DEBUG_SKIP_AUTH` on, a bearer token is ignored rather than enforced.

    `common/config.py` says the flag "will skip authorization when a token is not provided. note
    that if a token is provided, then auth will still occur" -- but `authorize_request` tests its
    `token` parameter, which `dependencies.py` never passes, and only reads the request's header
    on the following line. So the bypass wins and the policy engine is never asked.

    This test pins current behavior rather than endorsing it: if the check is reordered so the
    comment becomes true, this test is the one that should fail and be deleted.
    """

    class ExplodingArboristClient:
        async def auth_request(self, *args, **kwargs):
            raise AssertionError("the policy engine should not have been reached")

    allow_authz("docs")
    app.state.arborist_client = ExplodingArboristClient()

    create = client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "dimensions": 3, "vector_type": "vector"},
    )
    assert create.status_code == 200, create.text

    response = client.get("/vectorstore/collections/docs", headers={"Authorization": "Bearer some-token"})

    assert response.status_code == 200, response.text


def test_the_route_authorizes_its_own_action_on_its_own_collection(client, allow_authz, real_auth):
    """
    The read route asks the policy engine for `read` on the collection named in the path.

    Pins the mapping rather than just the outcome: a route that authorized the wrong action, or
    the wrong resource, would still return 403 here and look correct.
    """
    allow_authz("docs")
    arborist = real_auth(verdict=False)

    client.get("/vectorstore/collections/docs", headers={"Authorization": "Bearer some-token"})

    call = arborist.calls[0]
    assert call["resources"] == ["/vectorstore/collections/docs"]
    assert call["methods"] == "read"
    assert call["service"] == AUTHZ_SERVICE_NAME
    assert call["token"] == "some-token", "the caller's bearer credentials were not forwarded"


def test_a_granted_token_reaches_the_handler(client, allow_authz, real_auth):
    """
    When the policy engine allows the request, it proceeds to the handler as normal.

    The counterpart to the 403 case: without this, a gate that rejected everything would pass
    every other test in this file.
    """
    allow_authz("docs")

    # Created before the fake is installed, so this setup request takes the suite's usual
    # no-token bypass and only the read below is authorized for real.
    create = client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "dimensions": 3, "vector_type": "vector"},
    )
    assert create.status_code == 200, create.text

    arborist = real_auth(verdict=True)
    response = client.get("/vectorstore/collections/docs", headers={"Authorization": "Bearer some-token"})

    assert response.status_code == 200, response.text
    assert response.json()["collection_name"] == "docs"
    assert len(arborist.calls) == 1


def test_a_policy_engine_failure_is_not_reported_as_a_denial(client, allow_authz, app, monkeypatch):
    """
    If the policy engine itself errors, the request fails 500 rather than 403.

    A 403 would tell the caller they lack access when what actually happened is that nothing
    could determine whether they do -- and it would make an outage look like a permissions
    problem to everyone debugging it.
    """

    async def fake_get_user_id(token=None, request=None, authz_config=None):
        return "test-user"

    class BrokenArboristClient:
        async def auth_request(self, *args, **kwargs):
            raise RuntimeError("policy engine is down")

    monkeypatch.setattr(common_config, "DEBUG_SKIP_AUTH", False)
    monkeypatch.setattr(common_auth, "get_user_id", fake_get_user_id)
    allow_authz("docs")
    app.state.arborist_client = BrokenArboristClient()

    response = client.get("/vectorstore/collections/docs", headers={"Authorization": "Bearer some-token"})

    assert response.status_code == 500, response.text
