# File: trainer_app/native_dialogs.py
# Purpose: Native OS file/folder dialogs for the pywebview-hosted trainer app.
#
# Browser web apps cannot pick a folder or return real local paths, so these use
# the native dialog of the pywebview window we run inside. They fall back to
# tkinter when no webview window exists (e.g. plain-browser dev mode), and return
# None if neither is available or the user cancels.


def _from_webview(dialog_attr, **kwargs):
    """Try the pywebview native dialog; return a path str, or None if N/A/cancel.

    Returns the sentinel False if webview is unavailable (caller should fall back).
    """
    try:
        import webview
    except Exception:  # noqa: BLE001
        return False
    if not getattr(webview, "windows", None):
        return False
    dtype = getattr(webview, dialog_attr)
    res = webview.windows[0].create_file_dialog(dtype, **kwargs)
    if not res:
        return None
    return res[0] if isinstance(res, (list, tuple)) else res


def _from_tk(call):
    """Best-effort tkinter dialog fallback. Returns a path str or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            return call(filedialog) or None
        finally:
            root.destroy()
    except Exception:  # noqa: BLE001
        return None


def pick_folder():
    """Open a native folder-selection dialog. Returns a path str or None."""
    res = _from_webview("FOLDER_DIALOG")
    if res is not False:
        return res
    return _from_tk(lambda fd: fd.askdirectory())


def pick_open_file(file_types=("All files (*.*)",)):
    """Open a native open-file dialog. Returns a path str or None."""
    res = _from_webview("OPEN_DIALOG", file_types=file_types)
    if res is not False:
        return res
    return _from_tk(lambda fd: fd.askopenfilename())


def pick_save_file(default_name="model.nam", file_types=("NAM model (*.nam)",)):
    """Open a native save-as dialog. Returns a path str or None."""
    res = _from_webview("SAVE_DIALOG", save_filename=default_name,
                        file_types=file_types)
    if res is not False:
        return res
    return _from_tk(lambda fd: fd.asksaveasfilename(initialfile=default_name))
