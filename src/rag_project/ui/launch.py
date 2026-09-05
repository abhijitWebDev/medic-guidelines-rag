"""`rag-ui` -> streamlit run, with the app path resolved from the package."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from streamlit.web import cli as stcli

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), "--server.headless=true", *sys.argv[1:]]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
