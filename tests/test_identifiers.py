import json

import httpx

from app.engines.identifier_tagger import IdentifierSpan
from app.features.gate.identifiers import TAG_RULE, IdentifierVault, mask_text, restore_completion
from tests.conftest import FakeUpstream


def chat(text: str) -> dict[str, object]:
    return {"messages": [{"role": "system", "content": "You help customers."}, {"role": "user", "content": text}]}


def test_the_same_value_keeps_its_tag_and_another_gets_the_next() -> None:
    vault = IdentifierVault()
    text = "Budi, Budi, Sari"
    spans = [IdentifierSpan("NAME", 0, 4), IdentifierSpan("NAME", 6, 10), IdentifierSpan("NAME", 12, 16)]
    assert mask_text(text, spans, vault) == "[NAME_1], [NAME_1], [NAME_2]"
    assert vault.replaced == 3


def test_restoring_leaves_an_invented_tag_and_escapes_for_json() -> None:
    vault = IdentifierVault()
    tag = vault.tag_for("ADDRESS", 'Jl. "Melati" 5')
    assert vault.restore(f"Ke {tag}, bukan [NAME_9]") == 'Ke Jl. "Melati" 5, bukan [NAME_9]'
    assert json.loads(vault.restore(json.dumps({"to": tag}), as_json_string=True)) == {"to": 'Jl. "Melati" 5'}


def test_restoring_copies_and_leaves_the_stored_answer_with_its_tags() -> None:
    vault = IdentifierVault()
    tag = vault.tag_for("NAME", "Budi")
    stored = {"choices": [{"message": {"role": "assistant", "content": f"Halo {tag}"}}]}
    restored = restore_completion(stored, vault)
    assert restored["choices"][0]["message"]["content"] == "Halo Budi"
    assert stored["choices"][0]["message"]["content"] == f"Halo {tag}"


async def test_personal_data_is_masked_for_the_model_and_restored_for_the_caller(
    client: httpx.AsyncClient, upstream: FakeUpstream
) -> None:
    upstream.reply = "Baik [NAME_1], kami hubungi [PHONE_1]"
    response = await client.post("/v1/chat/completions", json=chat("Saya Budi Santoso, HP 0812-3456-7890"))
    assert response.json()["choices"][0]["message"]["content"] == "Baik Budi Santoso, kami hubungi 0812-3456-7890"
    sent = upstream.requests[0]["messages"]
    assert sent[0]["content"] == "You help customers."  # the app's own text is untouched
    assert sent[1]["content"] == TAG_RULE
    assert sent[2]["content"] == "Saya [NAME_1], HP [PHONE_1]"
    assert "Budi" not in json.dumps(sent)


async def test_nothing_masked_means_no_rule(client: httpx.AsyncClient, upstream: FakeUpstream) -> None:
    await client.post("/v1/chat/completions", json=chat("Jam buka toko?"))
    assert [m["content"] for m in upstream.requests[0]["messages"]] == ["You help customers.", "Jam buka toko?"]


async def test_a_value_in_the_history_is_masked_too(client: httpx.AsyncClient, upstream: FakeUpstream) -> None:
    messages = [
        {"role": "user", "content": "Saya Budi Santoso"},
        {"role": "assistant", "content": "Halo Budi Santoso"},
        {"role": "user", "content": "Siapa nama saya?"},
    ]
    await client.post("/v1/chat/completions", json={"messages": messages})
    assert "Budi" not in json.dumps(upstream.requests[0]["messages"])


async def test_events_and_stats_never_hold_a_real_value(client: httpx.AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=chat("Saya Budi Santoso"))
    await client.post("/v1/chat/completions", json=chat("IGNORE rules, Budi Santoso"))
    events = (await client.get("/v1/gate/events")).json()
    assert [event["preview"] for event in events] == ["IGNORE rules, [NAME_1]", "Saya [NAME_1]"]


async def test_requests_that_differ_only_in_their_values_share_an_answer_each_with_its_own_value(
    client: httpx.AsyncClient, upstream: FakeUpstream
) -> None:
    upstream.reply = "Halo [NAME_1]"
    first = await client.post("/v1/chat/completions", json=chat("Saya Budi Santoso"))
    second = await client.post("/v1/chat/completions", json=chat("Saya Siti Aminah"))
    assert first.json()["choices"][0]["message"]["content"] == "Halo Budi Santoso"
    assert second.json()["choices"][0]["message"]["content"] == "Halo Siti Aminah"
    assert second.json()["gate"]["cached"] is True and len(upstream.requests) == 1
