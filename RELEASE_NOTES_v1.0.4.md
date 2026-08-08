# Really-unREAL v1.0.4

v1.0.4 makes the desktop workflow match what users expect from the product.

## Clearer modes

- `빠른 Audit` is renamed to **빠른 진단** and explicitly described as a no-LLM data/timing health check.
- Local/NVIDIA replay generation is labeled **모델 테스트** so it is no longer mistaken for a live conversation.
- Historical Replay results are shown as a readable summary instead of raw developer JSON; full numeric JSON can still be saved.

## Live conversation

- Added **대화 시작** for Local LLM and NVIDIA NIM.
- Opens a messenger-like SIMULATION window for the selected Kakao direct-chat target.
- User-entered messages and model-generated messages are stored as `SIMULATION` only.
- A reply time is scheduled from historical relationship timing before the LLM is called.
- The UI shows a countdown such as `답장 예정 · 약 28초 후`.
- Pending reply/initiation events and simulation messages persist locally in SQLite across app restarts.
- `새 대화` clears simulation state only; imported REAL evidence is untouched.
- The live window never sends to a real messaging platform.

## Provider behavior

- Hosted NVIDIA use still requires explicit remote-private-context consent.
- The desktop NVIDIA request remains bounded to the v1.0.3 responsive timeout budget.

## Validation

The release workflow runs the full test suite, imports the live GUI modules, builds the Windows one-file executable, executes the packaged `--smoke` path, packages the portable ZIP, and publishes the release only if every gate succeeds.
