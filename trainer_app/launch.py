# File: trainer_app/launch.py
# Purpose: Start the Gradio app on localhost and show it in a native window.

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "launcher.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    try:
        import webview
        from trainer_app.app import build_app

        app = build_app()
        app.queue()  # enable live streaming of generator outputs (training log/progress)
        _, local_url, _ = app.launch(
            prevent_thread_lock=True, server_port=7860,
            inbrowser=False, show_error=True,
        )
        logging.info("Gradio running at %s", local_url)
        webview.create_window(
            "NAM Parametric Trainer", local_url, width=1100, height=820)
        webview.start()
    except Exception:
        logging.exception("Launcher failed")
        raise


if __name__ == "__main__":
    main()
