from __future__ import annotations

import sys
from zoneinfo import ZoneInfo

from backend import __version__


def bundle_smoke() -> None:
    """Verify resources that must survive PyInstaller bundling."""

    ZoneInfo("UTC")
    ZoneInfo("Asia/Seoul")
    import backend.gui_support  # noqa: F401
    import backend.ingest.instagram  # noqa: F401
    import backend.replay_generation  # noqa: F401


def main() -> None:
    if "--smoke" in sys.argv:
        bundle_smoke()
        return

    from backend.gui import ReallyUnrealApp

    app = ReallyUnrealApp()
    app.mainloop()


if __name__ == "__main__":
    main()
