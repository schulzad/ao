import os
import sys


def _ensure_repo_root_on_path():
	repo_root = os.path.dirname(os.path.dirname(__file__))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)


_ensure_repo_root_on_path()


