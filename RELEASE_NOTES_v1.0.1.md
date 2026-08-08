# Really-unREAL v1.0.1

v1.0.1 is the usability release for the v1.0 simulation core.

## New

- Small local desktop GUI (`Really-unREAL.exe` on Windows builds).
- KakaoTalk ZIP file picker.
- Automatic self-display-name suggestion with explicit user confirmation.
- Direct-chat target selection ordered by available target evidence.
- One-click **Quick Audit** that needs no LLM or API key.
- Optional local OpenAI-compatible model generation from the GUI.
- Optional hosted NVIDIA NIM generation behind explicit remote private-context consent.
- Aggregate-only results panel and JSON export.
- `really-unreal` GUI entry point when installed as a Python package.
- Windows x64 portable PyInstaller build artifact.
- Korean/English-friendly quick-start documentation.

## Privacy

The GUI keeps the v1 core boundaries:

- raw archives remain local;
- the quick path does not require uploading an identity map;
- remote model endpoints remain blocked unless the user explicitly allows private-context transmission;
- API keys are not written into result files;
- the GUI does not print raw private chat text in its result panel;
- no real messaging platform is controlled or auto-sent to.

## Scope

The v1.0.1 GUI intentionally focuses on the simplest safe first-run path: **Kakao direct-chat PERSON replay**.

Advanced workflows remain available through the CLI, including Instagram fusion, SELF twin, dense retrieval, source ablation, custom identity maps, Shadow Simulation, and the persistent live runtime.

## Build

The Windows build workflow must pass the test suite, import the GUI entry point, build the one-file executable, and package the portable ZIP before the build status is considered successful.
