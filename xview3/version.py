__version__ = "2021.05.28"

__all__ = ["get_version", "get_version_with_tag", "git_commit_tag"]


def get_version():
    return __version__


def git_commit_tag():
    return ""


def get_version_with_tag():
    return f"{get_version()}.{git_commit_tag()}"
