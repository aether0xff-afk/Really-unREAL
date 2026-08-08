from __future__ import annotations

import sys
from zoneinfo import ZoneInfo

from backend import __version__


def bundle_smoke() -> None:
    """Verify resources and 1.2 runtime modules survive PyInstaller bundling."""

    ZoneInfo("UTC")
    ZoneInfo("Asia/Seoul")
    import backend.gui_support  # noqa: F401
    import backend.gui_runtime  # noqa: F401
    import backend.gui_live  # noqa: F401
    import backend.gui_chat_window  # noqa: F401
    import backend.gui_responsive  # noqa: F401
    import backend.live_behavior  # noqa: F401
    import backend.live_timing  # noqa: F401
    import backend.simulation.store  # noqa: F401
    import backend.simulation.runtime  # noqa: F401
    import backend.generation_guard  # noqa: F401
    import backend.persona.style_fingerprint  # noqa: F401
    import backend.providers.errors  # noqa: F401
    import backend.ingest.instagram  # noqa: F401
    import backend.replay_generation  # noqa: F401


def main() -> None:
    if "--smoke" in sys.argv:
        bundle_smoke()
        return

    from backend.gui_responsive import ResponsiveReallyUnrealApp

    app = ResponsiveReallyUnrealApp()
    app.mainloop()


if __name__ == "__main__":
    main()
