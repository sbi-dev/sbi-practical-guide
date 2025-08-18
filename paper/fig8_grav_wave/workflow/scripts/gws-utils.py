import subprocess
from datetime import datetime

def git_available() -> bool:

    value = True
    try:
        _ = subprocess.check_output(['git', '--version']).decode('ascii').strip()
    except Exception as ex:
        value = False
        print(f"gws-utils.py: WARNING git likely not available on this system {ex}")

    return value


def get_git_revision_short_hash() -> str:
    if git_available():
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    else:
        today = datetime.now().strftime("%y%m%d-%H:%M:%S")
        return f"git-unavailable-{today}"


def get_git_describe() -> str:
    if git_available():
        return subprocess.check_output(['git', 'describe', '--always', '--dirty=*']).decode('ascii').strip()
    else:
        today = datetime.now().strftime("%y%m%d-%H:%M:%S")
        return f"git-unavailable-{today}"
