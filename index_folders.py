"""
Indexes the folders in KNOWN_FOLDERS OUTSIDE of the voice assistant.

Usage (run with the voice assistant CLOSED, so RAM/VRAM are fully free):
    python index_folders.py                    -> all folders
    python index_folders.py university          -> a single alias
    python index_folders.py "project code"       -> alias with spaces, quoted

Can be interrupted with Ctrl+C and rerun: it resumes where it left off.
"""
import sys
import logging

from rag_engine import KNOWN_FOLDERS, sync_folder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    aliases = sys.argv[1:] or list(KNOWN_FOLDERS.keys())

    for alias in aliases:
        if alias not in KNOWN_FOLDERS:
            logger.error(f"Unknown alias: '{alias}'. Available: {', '.join(KNOWN_FOLDERS)}")
            continue
        logger.info(f"\n=== Indexing: {alias} ===")
        try:
            sync_folder(alias)
        except KeyboardInterrupt:
            logger.info("Interrupted. Already-indexed files remain saved: rerun to continue.")
            break


if __name__ == "__main__":
    main()
