# Really-unREAL 1.2 GUI Quick Start

The desktop UI lets you load KakaoTalk exports and use the replay/live simulator without editing identity JSON by hand.

## Windows portable build

1. Download `Really-unREAL-v1.2.0-Windows-x64.zip` from the GitHub Release.
2. Extract it and run `Really-unREAL.exe`.
3. SmartScreen may warn because the community binary is unsigned.

The app never sends a message to KakaoTalk, Instagram, or another real messaging platform.

## 1. Load KakaoTalk ZIPs

Choose **여러 ZIP 선택**. Ctrl selects individual files and Shift selects a range. One selected ZIP may also be an outer bundle containing several exports. Exact duplicate conversations are de-duplicated.

Confirm **내 이름**, then choose **대화 상대**.

## 2. 빠른 진단

No LLM is called. This checks parsing, self/target mapping, replay availability, and timing-model data.

## 3. 모델 테스트

**모델 테스트 · Local LLM** and **모델 테스트 · NVIDIA NIM** hide real historical continuations and evaluate the reconstruction using only older evidence.

The UI keeps content overlap, endings, message splitting, expression behavior and timing separate. Timing is evaluated independently of provider generation success. Mean metrics use small-sample Student-t intervals and binary match rates use Wilson intervals.

The NVIDIA default of **3 cases is only a smoke check**. Use roughly 10–20+ held-out cases for comparisons when practical.

### Local model

Default OpenAI-compatible endpoint:

```text
http://127.0.0.1:1234/v1
```

### NVIDIA NIM

Default hosted model:

```text
nvidia/nemotron-3-ultra-550b-a55b
```

Hosted inference sends cutoff-safe private prompt context to NVIDIA. It therefore requires the explicit remote-context consent checkbox. The API key is not written into benchmark result files.

## 4. 대화 시작 — Live SIMULATION

1. Your message is stored as `SIMULATION` only.
2. The relationship behavior policy first chooses **REPLY or WAIT**. WAIT means no reply event is created.
3. If a reply exists, an inferred **READ** event and a separate **REPLY** event can occur at different times.
4. Only message actions call the LLM; READ never calls it.
5. After a target message, the simulator separately considers a same-session **FOLLOW_UP**, a later new-session **INITIATE**, or silence.

### Rapid user bubbles

Sending `야` → `뭐함` → `ㅋㅋ` quickly no longer re-rolls the person's reply timer each time. A future unclaimed reply keeps its sampled clock; only a small settle floor prevents generation in the middle of rapid typing.

If an earlier bubble was already marked `읽음 추정` and you send another before the reply, the later bubble gets its own inferred READ opportunity.

### Read indicator

`안읽음 추정` / `읽음 추정 · HH:MM` are **simulation inference**, not real KakaoTalk read receipts. Kakao exports do not contain true read timestamps.

### Provider failures

The person's modeled behavior and the provider delivery clock are separate.

- temporary 429/5xx/timeout/network failure → exact event becomes `RETRY`; original behavior time stays immutable;
- credential/config/invalid response failure → exact event becomes `BLOCKED`;
- retries cannot see messages after the original modeled behavior cutoff;
- a later user message cannot cancel a generation that was already atomically claimed;
- stale claims recover after restart.

### Multiple windows / crashes

Due events are atomically claimed in SQLite. A second app/window cannot generate the same event simultaneously. Generated messages and `PROCESSED` completion are committed together. Resetting while a generation is in flight prevents stale output from reappearing in the new simulation session.

### Message splitting

Multi-bubble generated responses use observed REAL target-burst spacing when available rather than a fake fixed one-second gap.

## Privacy boundaries

- imported evidence is `REAL`;
- user-entered and generated live messages are `SIMULATION`;
- SIMULATION never becomes asserted REAL history;
- raw archives remain local unless you explicitly use a hosted inference route;
- hosted/private-context routes require explicit consent;
- no real messaging-platform send API is called;
- live style exemplars are private prompt-time evidence and are not written to public result JSON/release artifacts.

## Important interpretation limit

Really-unREAL 1.2 still uses a general language model conditioned by person-specific observable evidence. It does not claim neural fine-tuning into a person or reconstruction of hidden mental state.

## Build from source

```bash
python -m pip install -e '.[dev,build]'
pytest -q
pyinstaller --noconfirm --clean --onefile --windowed --name Really-unREAL backend/gui_entry.py
./dist/Really-unREAL.exe --smoke
```
