# Really-unREAL v1.0.3

v1.0.3 fixes the desktop NVIDIA NIM run that could appear stuck for several minutes with no progress information.

## Fixed

- Added per-case progress in the desktop GUI.
- Added elapsed time display during generation.
- Added a Cancel button for hosted/local generation runs.
- Reduced the desktop NVIDIA request budget from the previous worst-case 90s × 3 transport attempts × 2 format attempts to a bounded 45s × 1 × 1 per in-flight case.
- A failed generation case no longer discards already completed cases; failures are counted and the run continues.
- Cancellation is cooperative and takes effect after the current request returns or reaches its bounded timeout.
- NVIDIA NIM now defaults to 3 evaluation cases when switching providers; users may still increase the case count manually.

## Privacy

The patch does not change the privacy model. Hosted NVIDIA generation still requires explicit remote-private-context consent and raw archives remain local.
