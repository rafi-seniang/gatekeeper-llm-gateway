# 0004. Model masking with restored tags, and Jev checks for agents

Date: 1 Nov 2026. Status: accepted.

## Context

Identifier patterns missed every name and address and leaked 80% of personal values on a public benchmark, and more patterns would
only move the gap. Masking also lost the values for good, so an answer could not address the user by name. The guard read tool
results with a question about overriding instructions, and caught none of 911 indirect attacks. Nothing looked at what the model
wanted to do next.

## Decision

- A local ONNX tagger finds personal data, on the CPU, before anything leaves the machine. No patterns remain. Only its weights are
  used, pinned to one revision, and spans are decoded in this repository.
- Values become tags kept in a per-request vault, and the answer gets them back. The history is masked too, because it holds restored
  values. The cache key and the stored answer use tags, so the cache holds no real value, and requests differing only in values share
  an answer.
- With `AGENT_GUARD_ENABLED`, new tool results go through four Jev questions in one request, and each tool call in the reply goes through
  two more before it is returned. A stopped call is a 403, like any block.
- Jev is pinned to one version, because the weights and thresholds belong to it.

## Consequences

- Up to two more Jev calls per agent turn, about 240 ms each. Plain chat with no tools pays none of this.
- The gate needs a model file and 13 to 45 ms of CPU per text.
- Wrongly masked words reach the model as tags and can weaken an answer.
- A new Jev version means fitting the weights and reading the thresholds again.
- Restoring means a reply can carry real values wherever the model writes a tag. A planted instruction could put a tag in a link, and the
  gate does not read replies for that yet.
