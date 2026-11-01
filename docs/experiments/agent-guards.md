# Model masking and agent guards (1 Nov 2026)

Goal: replace the identifier patterns with a model, and find where Jev earns a place beyond the input guard. Settings were
chosen on a tuning part and reported on held-out data. The runs, the data handling and the ideas that failed are in the Lab
repository's `experiments/pii-masking` and `experiments/jev-gateway`.

## Masking

IndoPII-Bench test (2,591 values, 480 documents). Leak is the share of values not fully covered by a mask.

| System | Leak | Overmask | Clean documents touched of 86 |
|---|---|---|---|
| The earlier regex | 79.8% | 4.2% | 5 |
| **IndoPII Base tagger, threshold 0.5 (now)** | **4.1%** | **2.9%** | 17 |
| The tagger and the regex together | 3.9% | 3.4% | 20 |
| DeepSeek Flash alone | 12.0% | 6.3% | 15 |

On email, NIK, phone and card only, the tagger leaks 6 of 491 and the regex 35. Median 13 ms on a short message and 45 ms on a
394-character document, on 4 CPU threads. On real sentences the leak is 4.0%. On 300 English texts it is 31.2%, because foreign
IDs and postcodes are weak, while email (2 of 93) and phone (2 of 103) hold up. The benchmark and the model share one author and
the main set is written by LLMs. An independent run came within 2 points of the model card's F1.

Tags restored after the model kept 168 of 219 values, against 169 for a raw-text answer. A blind judge preferred the raw answer
24 to 14 with 2 ties, where a placebo gave 15, 13 and 12, so answers read somewhat worse. Pseudonyms did worse (33 to 4) and left
stand-ins in 9 of 40 answers.

## Tool-result guard

911 held-out attacks from BIPIA and InjecAgent, each with a clean twin, plus 350 real RAGTruth passages that hold no attack.

| Guard | Caught | Clean flagged of 564 |
|---|---|---|
| The input question at 0.98 | 0 | 0 |
| DeBERTa at 0.5 | 269 | 62 of 214 twins |
| **Four questions in one request, 0.9428 (now)** | **576 (63%)** | **0** |
| The same at 0.8846 | 689 (76%) | 3 |

An earlier fit without real documents claimed 91%, and low thresholds on the input question flagged 167 of 500 real documents.
Real documents have to be in the fit.

## Tool-call check

The state holds the masked request and the call. A call is stopped when `asked` is below 0.5 or `scope` is 0.7 or more.

| Set | Injected or swapped calls stopped | Legitimate calls stopped |
|---|---|---|
| InjecAgent | 300 of 300 | 1 of 600 |
| 34 hand-written cases, masked by the real tagger (`scripts/eval_agent_guard.py`) | 13 of 14 | 0 of 20 |

A `scope` threshold of 0.3 looked perfect on the hand-written cases and stopped 66 of 600 legitimate calls on InjecAgent. InjecAgent's
attacks are mostly unrelated to the user's task, which makes them easier. The miss swaps a bank account through a tool result.

## Cost and time

One Jev request takes about 240 ms with 1 or 20 questions, and each question adds about 56 input tokens, at $0.042 per million.

## Not shipped

- **Several questions on the input.** Six answers caught 402 of 1,568 held-out jailbreaks with 1 of 2,432 wrong blocks, against 211
  today. At that threshold a fake end of state dropped it from 110 to 27 of 113 attacks, and a JSON state only recovered it to 41.
- **Grounding check.** F1 81.3 on RAGTruth with no training, against 79.2 for LettuceDetect large. It needs a way to know which
  message is the source.
- **Verifying semantic cache hits.** Level with Redis's trained cross-encoder overall (AUC 0.850 and 0.848), behind on PAWS, ahead on
  40 hand-written Indonesian pairs.
- **Jev for general content safety.** Level with the local Qwen3Guard, which keeps text on the machine.
