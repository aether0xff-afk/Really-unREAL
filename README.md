# Really-unREAL 1.2.0

A local-first conversation-behavior digital-twin framework grounded in real message history.

Really-unREAL does **not** claim to reconstruct hidden thoughts, feelings, attraction, intent, or a person's mind. It models narrower observable behavior: **whether someone stays silent, whether/when they read or reply, whether they follow up or start a new session, how they split messages, and how they write.**

> Make the most plausible behavior from the evidence, not the most entertaining behavior.

## Quick start — Windows GUI

1. Download `Really-unREAL-v1.2.0-Windows-x64.zip` from the GitHub Release.
2. Extract it and run `Really-unREAL.exe`.
3. Choose one or many KakaoTalk export ZIP files.
4. Confirm the suggested **내 이름**.
5. Choose a direct-chat **대화 상대**.
6. Run **빠른 진단** to check the imported data without an LLM.
7. Optionally run **모델 테스트** on held-out historical replies.
8. Choose Local LLM or NVIDIA NIM and press **대화 시작**.

The app never sends messages to KakaoTalk, Instagram, or another real messaging platform.

## 1.2 Live behavior model

1.2 fixes the main structural limitation of 1.1: a user message no longer implies that a reply must eventually exist.

```text
current observable context
        |
        v
   REPLY or WAIT?  ---------------------- WAIT -> silence
        |
      REPLY
        |
        +---- inferred READ event (no LLM)
        |
        +---- reply timing
        v
   generation only when due
        |
        v
  FOLLOW_UP / INITIATE / WAIT
```

The runtime now distinguishes:

- `WAIT` — no message behavior is scheduled;
- `READ` — simulation-only inferred read state, never a real Kakao receipt;
- `REPLY` — response to the counterpart;
- `FOLLOW_UP` — another target message inside an active session;
- `INITIATE` — a new-session start after a long idle gap.

### REPLY vs WAIT

A relationship-specific binary behavior policy is fitted from complete historical counterpart-message bursts. Cases where the counterpart sends another burst before the target answers provide conservative silence/WAIT evidence. The final export edge is treated as censored rather than automatically labeled as an ignore.

### READ is separate from REPLY

Kakao exports do not contain ground-truth read timestamps. 1.2 therefore keeps READ explicitly labeled as **SIMULATION inference** and schedules it separately from REPLY. A message can show `읽음 추정` before the modeled reply arrives. Provider outages cannot move an already-modeled read/reply behavior time.

### FOLLOW_UP is not INITIATE

1.1 used same-sender continuation as an initiation proxy. 1.2 separates same-session `FOLLOW_UP` from a long-gap new-session `INITIATE`. Long-gap target messages after the counterpart remain action-role ambiguous instead of being force-labeled.

## Timing

Live timing can use observable context including:

- relationship and action role;
- time of day and weekday/weekend;
- recent 15-minute activity;
- previous visible-message gap;
- **time since the last visible message**;
- question / very-short / statement type, including common Korean questions without `?`.

The richer discrete hazard model is used only when held-out validation earns it. A tiny empirical cell no longer overrides a validated hazard model. Sparse histories back off conservatively.

Kakao minute timestamps remain feasible intervals rather than fake exact seconds.

## User and target message bursts

Rapid user bubbles no longer re-sample and reset the reply clock on every press of Send. A pending unclaimed reply keeps its sampled behavior time, with only a small input-settle floor to avoid firing mid-typing.

Generated multi-bubble replies no longer use a fake fixed `+1 second` spacing. Internal burst gaps are sampled from observed REAL target bursts when available.

## Race/crash safety

SQLite dispatch now uses an explicit event lifecycle:

```text
PENDING / RETRY
      |
      v
   CLAIMED  <--- atomic claim; another window/process cannot claim it
      |
      +-- provider temporary failure -> RETRY
      +-- permanent/config error ----> BLOCKED
      +-- success -------------------> message insert + PROCESSED atomically
```

- provider retry time remains separate from immutable modeled behavior time;
- a later user message cannot cancel an already claimed generation;
- retries cannot see context after the original modeled behavior cutoff;
- stale claims are recovered after restart;
- resetting a chat while generation is in flight prevents stale output from reappearing;
- generated messages and event completion are committed in one transaction.

## Person-specific generation

Generation combines current visible conversation, relationship-focused style fingerprint, burst profile, cutoff-safe topic/event memory, historical-situation retrieval, and at most two older REAL style exemplars in Live mode.

The anti-copy guard now checks both whole-output similarity and historical-phrase containment. If repeated generation remains too close to an old long exemplar, the final attempt removes raw historical response wording entirely and generates from style/structure evidence only. Common short expressions such as `ㅇㅇ`, `ㄴㄴ`, or `ㅋㅋ` may still recur naturally.

## Evaluation

Historical Replay remains chronological and future-leakage-safe. Timing evaluation is now independent of provider generation success: an NVIDIA/local failure cannot silently remove a timing case from the metric.

For uncertainty:

- mean-like content metrics use Student-t intervals for small samples;
- binary match rates use Wilson intervals;
- the default NVIDIA 3-case run remains a smoke check, not a fidelity estimate.

Use 10–20+ held-out cases for comparisons when practical.

## Privacy model

```text
REAL        observations imported from real records
SIMULATION  generated events and user-entered simulation messages
```

- Raw archives and `identity.local.json` are not release content.
- SIMULATION output is never promoted into asserted REAL history.
- Loopback model endpoints are local by default.
- Non-loopback private-context transmission requires explicit consent.
- API keys are not written to evaluation JSON or release artifacts.
- Live style exemplars are private prompt-time evidence.
- No real messaging-platform send API exists in the core.
- `읽음 추정` is a simulation state, not a real read receipt.

## Important limits

- Human communication is stochastic; replay similarity is not identity reconstruction.
- The base LLM is still a general language model conditioned by person-specific observable evidence, not a neural fine-tune into a real person.
- Read timing is latent/inferred because exports do not provide true read receipts.
- True new-session initiations are sparse and harder to model than replies.
- Sparse relationships require broader statistical backoff.
- Hosted inference still depends on provider availability, though outages no longer alter modeled behavior decisions.
- Windows portable binaries are unsigned community builds, so SmartScreen may warn.

## Build from source

Python 3.11+:

```bash
python -m pip install -e '.[dev,build]'
pytest -q
pyinstaller --noconfirm --clean --onefile --windowed --name Really-unREAL backend/gui_entry.py
./dist/Really-unREAL.exe --smoke
```

Relevant docs:

- `docs/QUICKSTART_GUI.md`
- `docs/HISTORICAL_REPLAY.md`
- `docs/TEMPORAL_HAZARD.md`
- `docs/CUTOFF_RAG.md`
- `docs/SELF_TWIN.md`

## Release history

- 1.0 — leakage-safe Historical Replay foundation
- 1.0.4 — persistent desktop Live Simulation
- 1.0.5 — stochastic observed timing instead of fixed median
- 1.1 — provider resilience, context timing, stronger person-specific generation
- 1.1.1 — inferred read/unread UI
- **1.2.0 — true WAIT, READ/REPLY/FOLLOW_UP/INITIATE separation, burst handling, atomic event runtime, evaluation/anti-copy hardening**
