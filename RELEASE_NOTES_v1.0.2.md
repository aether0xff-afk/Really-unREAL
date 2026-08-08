# Really-unREAL v1.0.2

v1.0.2 is a small usability patch focused on KakaoTalk import.

## New

- Select multiple KakaoTalk ZIP files in one file-picker operation.
- Combine conversations from all selected archives into one local analysis set.
- Each selected ZIP may be either a single-chat export or an outer bundle containing several chat ZIPs.
- Exact duplicate conversations are removed automatically to prevent accidental double-counting.
- Repeated selections of the same archive path are ignored.
- The GUI shows how many ZIPs were selected and how many conversations were loaded.
- The quick-start guide now documents Ctrl/Shift multi-selection on Windows.

## Compatibility

- Selecting exactly one ZIP works as before.
- The existing outer-bundle ZIP format remains supported.
- No private raw chat data is added to builds, Actions artifacts, or the repository.

## Validation

The release workflow runs the test suite, imports the desktop GUI, builds the Windows one-file executable, executes the packaged `--smoke` path, and only then publishes the portable ZIP release asset.
