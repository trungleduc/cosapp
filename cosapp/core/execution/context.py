import platform


def is_running_windows() -> bool:
    return platform.system() == "Windows"


def is_running_mac() -> bool:
    return platform.system() == "Darwin"


def is_running_linux() -> bool:
    return not (platform.system() in ("Windows", "Darwin"))


is_fork_available = is_running_linux
