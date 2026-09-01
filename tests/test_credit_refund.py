"""A failed API call must not leave a charge behind.

Measured against the provider's own dashboard on 2026-09-01: ten profile calls,
nine of which returned HTTP 500, moved the real balance by exactly one credit.
Failures are not billed. The client charges before the request so a runaway loop
cannot outrun the budget guard, which is the right ordering — but it means a
failure leaves a phantom charge unless it is given back.

Left uncorrected this drifts one way only: the ledger reports less credit than
the account actually holds, and eventually refuses to run while credits remain.
On a budget that cannot be topped up, that is throwing money away.
"""
from __future__ import annotations

import pytest
import requests

from cdc.collect.instagram import CreditLedger, CreditsExhausted, InstagramClient


def _ledger(spent=0, budget=100):
    return CreditLedger(budget_total=budget, spent_total=spent)


def test_refund_returns_the_credit():
    led = _ledger(spent=10)
    led.charge(1)
    assert led.spent_total == 11
    led.refund(1)
    assert led.spent_total == 10


def test_refund_never_goes_below_zero():
    led = _ledger(spent=0)
    led.refund(5)
    assert led.spent_total == 0


def test_refund_reverses_the_daily_tally():
    led = _ledger(spent=0)
    led.charge(1)
    day = next(iter(led.by_day))
    assert led.by_day[day] == 1
    led.refund(1)
    assert led.by_day[day] == 0


def test_server_error_refunds_the_charge(monkeypatch):
    """The exact scenario the screening run hit: repeated HTTP 500."""
    led = _ledger(spent=0)
    client = InstagramClient(api_key="k", ledger=led)

    class Resp:
        status_code = 500
        def raise_for_status(self):
            raise requests.HTTPError("500 Server Error")

    monkeypatch.setattr(client.session, "get", lambda *a, **k: Resp())
    monkeypatch.setattr("time.sleep", lambda *_: None)

    with pytest.raises(requests.HTTPError):
        client._get("/v1/instagram/profile", {"handle": "x"})
    assert led.spent_total == 0, "a 500 was billed locally but not by the provider"


def test_network_error_refunds_the_charge(monkeypatch):
    led = _ledger(spent=0)
    client = InstagramClient(api_key="k", ledger=led)

    def boom(*a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(client.session, "get", boom)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    with pytest.raises(requests.ConnectionError):
        client._get("/v1/instagram/profile", {"handle": "x"})
    assert led.spent_total == 0, "a request that never completed cannot be billed"


def test_successful_call_keeps_the_charge(monkeypatch):
    led = _ledger(spent=0)
    client = InstagramClient(api_key="k", ledger=led)

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"data": {"user": {}}}

    monkeypatch.setattr(client.session, "get", lambda *a, **k: Resp())
    client._get("/v1/instagram/profile", {"handle": "x"})
    assert led.spent_total == 1


def test_budget_guard_still_fires_before_the_request(monkeypatch):
    """Refunding must not weaken the guard that stops runaway spending."""
    led = _ledger(spent=100, budget=100)
    client = InstagramClient(api_key="k", ledger=led)
    monkeypatch.setattr(client.session, "get",
                        lambda *a, **k: pytest.fail("must not reach the network"))
    with pytest.raises(CreditsExhausted):
        client._get("/v1/instagram/profile", {"handle": "x"})
