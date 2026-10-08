"""
Local semantic search over known folders: indexes PDF/Markdown/text/Python
files into ChromaDB (multilingual embeddings) and answers natural-language
queries with the most relevant chunks, citing source file and page.

Designed to be driven by voice: folder names are matched against spoken
text with fuzzy matching to tolerate speech-to-text transcription errors.
"""
import os
import gc
import json
import time
import hashlib
import logging
import chromadb
from chromadb.utils import embedding_functions
from difflib import get_close_matches
from pypdf import PdfReader

try:
    import psutil
    _process = psutil.Process(os.getpid())
except ImportError:
    _process = None  # still works without it, just without RAM monitoring

# pypdf logs a warning per page for certain fonts; these are not errors.
logging.getLogger("pypdf").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ============================ CONFIGURATION ============================

# Spoken alias -> real folder path. REPLACE WITH YOUR OWN FOLDERS.
KNOWN_FOLDERS = {
    "test": r"C:\path\to\a\test\folder"
}

# Words that, together with a recognized folder alias, trigger a file search.
SEARCH_TRIGGER_WORDS = {"cerca", "trova", "cercami", "trovami", "cartella", "file"}

SUPPORTED_EXTENSIONS = (".py", ".md", ".txt", ".pdf")

# Folders that are never scanned. Hidden folders (starting with ".") and any
# virtualenv (detected via pyvenv.cfg, regardless of its name) are also skipped.
EXCLUDED_DIRS = {
    "venv", ".venv", "env", "site-packages", "dist-packages", "node_modules",
    "__pycache__", "chroma_db", "folder_cache_db", "build", "dist",
}

MAX_FILE_SIZE_MB = 50        # files larger than this are skipped
MAX_CHUNKS_PER_FILE = 6000   # safety cap per single file

# The multilingual embedding model truncates input at 128 tokens (~500 chars):
# longer chunks would only be "seen" partially. 500/100 keeps full coverage.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

UPSERT_BATCH_SIZE = 128      # chunks embedded and saved per batch (keeps RAM low)
N_RESULTS = 4                # chunks returned per search

# Safety guard: if RAM usage grows more than this during indexing, stop
# (already-indexed files remain saved) instead of risking a system freeze.
MAX_INDEX_RAM_GROWTH_GB = 6.0

# Live (voice-triggered) indexing handles at most this many new/changed files;
# beyond that, the user is told to run index_folders.py offline instead.
AUTO_INDEX_MAX_FILES = 10

MTIME_CHECK_COOLDOWN_SECONDS = 10

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
STATE_FILE = "./folder_index_state_v2.json"

# ========================================================================

_embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path="./folder_cache_db")


def _load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state():
    # Atomic write: if the process is interrupted mid-write, the state file
    # is never left corrupted.
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_state, f)
    os.replace(tmp_path, STATE_FILE)


# collection_name -> {file_path: signature}. Tracks, file by file, what's indexed.
_state = _load_state()
_last_checked = {}   # folder_path -> timestamp of the last check (in-memory only)
_last_notice = {}    # folder_path -> last generated notice


def _rss_gb() -> float | None:
    return _process.memory_info().rss / (1024 ** 3) if _process else None


def _collection_name_for(folder_path: str) -> str:
    # Model name and chunk size are part of the key: different embeddings
    # aren't comparable, so changing them creates a fresh index instead of
    # mixing incompatible vectors.
    key = f"{folder_path}|{EMBEDDING_MODEL_NAME}|{CHUNK_SIZE}|{CHUNK_OVERLAP}"
    return "folder_" + hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------- Folder recognition ----------------------------

def resolve_folder_alias(spoken_text: str, cutoff: float = 0.6) -> str | None:
    """Finds a known alias INSIDE a spoken sentence (not a full-sentence match).
    Tries an exact substring match first, then a fuzzy word-by-word match to
    tolerate small Whisper transcription errors (e.g. "progetto codici"
    instead of "progetto codice"). Returns the ALIAS, not the path."""
    spoken_text = spoken_text.lower().strip()

    for alias in KNOWN_FOLDERS:
        if alias in spoken_text:
            return alias

    words = spoken_text.split()
    candidates = words + [" ".join(pair) for pair in zip(words, words[1:])]
    for candidate in candidates:
        match = get_close_matches(candidate, KNOWN_FOLDERS.keys(), n=1, cutoff=cutoff)
        if match:
            return match[0]

    return None


def should_search_files(spoken_text: str) -> tuple[bool, str | None]:
    """(True, alias) if the sentence contains both a recognized folder alias
    AND a trigger word. Naming a folder without asking to search in it is
    not enough to activate search mode."""
    spoken_lower = spoken_text.lower()
    alias = resolve_folder_alias(spoken_lower)
    if alias is None:
        return False, None

    has_trigger = any(word in spoken_lower for word in SEARCH_TRIGGER_WORDS)
    return (True, alias) if has_trigger else (False, None)


def _clean_query(query: str, folder_alias: str) -> str:
    """Strips the folder alias and command words, leaving only what to search for."""
    cleaned = query.lower().replace(folder_alias, " ")
    words = [w for w in cleaned.split() if w.strip(",.?!") not in SEARCH_TRIGGER_WORDS]
    return " ".join(words).strip() or query


# ---------------------------- File scanning ----------------------------

def _iter_candidate_files(folder_path: str, skipped: list | None = None):
    for root, dirs, files in os.walk(folder_path):
        if "pyvenv.cfg" in files:      # it's a virtualenv, whatever it's named
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]

        for filename in files:
            if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            filepath = os.path.join(root, filename)
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
            except OSError:
                continue
            if size_mb > MAX_FILE_SIZE_MB:
                if skipped is not None:
                    skipped.append(f"{filename} ({size_mb:.0f} MB)")
                continue
            yield filepath


def _file_signature(filepath: str) -> str | None:
    try:
        st = os.stat(filepath)
        return f"{int(st.st_mtime)}:{st.st_size}"
    except OSError:
        return None


def _file_size(filepath: str) -> int:
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0


def _iter_chunks(text: str):
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(text), step):
        chunk = text[i:i + CHUNK_SIZE]
        if chunk.strip():
            yield chunk


def _iter_file_chunks(filepath: str):
    """Yields (chunk, page_number) one at a time, so per-file memory stays
    bounded. page_number is None for non-PDF files."""
    if filepath.lower().endswith(".pdf"):
        try:
            reader = PdfReader(filepath)
            total_pages = len(reader.pages)
        except Exception:
            return
        for page_number, page in enumerate(reader.pages, start=1):
            if page_number % 50 == 0:
                logger.info(f"  ...page {page_number}/{total_pages}")
            try:
                page_text = page.extract_text() or ""
            except Exception:
                continue
            for chunk in _iter_chunks(page_text):
                yield chunk, page_number
    else:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return
        for chunk in _iter_chunks(content):
            yield chunk, None


# ---------------------------- Indexing ----------------------------

def _index_file(collection, filepath: str) -> int:
    """Indexes a single file in small batches. Returns the number of chunks saved."""
    filename = os.path.basename(filepath)
    path_hash = hashlib.md5(filepath.encode()).hexdigest()[:16]

    docs, metadatas, ids = [], [], []
    total = 0

    for i, (chunk, page) in enumerate(_iter_file_chunks(filepath)):
        if i >= MAX_CHUNKS_PER_FILE:
            logger.warning(f"  reached the {MAX_CHUNKS_PER_FILE}-chunk cap: the rest of the file is skipped")
            break

        metadata = {"source": filename, "path": filepath}
        if page is not None:
            metadata["page"] = page
        docs.append(chunk)
        metadatas.append(metadata)
        ids.append(f"{path_hash}_{i}")
        total += 1

        if len(docs) >= UPSERT_BATCH_SIZE:
            collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
            docs, metadatas, ids = [], [], []

    if docs:
        collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
    return total


def _plan_sync(folder_path: str):
    """Compares files on disk with what's already indexed.
    Returns (collection_name, current, removed, to_index)."""
    collection_name = _collection_name_for(folder_path)
    known = _state.get(collection_name, {})

    current = {}
    for path in _iter_candidate_files(folder_path):
        signature = _file_signature(path)
        if signature:
            current[path] = signature

    removed = [p for p in known if p not in current]
    to_index = [p for p, sig in current.items() if known.get(p) != sig]
    to_index.sort(key=_file_size)   # small files first: quick wins before the big ones
    return collection_name, current, removed, to_index


def sync_folder(folder_alias: str) -> bool:
    """
    Syncs the index to the folder: indexes new/changed files, removes deleted
    ones. Saves progress file by file, so an interrupted run resumes where it
    left off. Returns True on completion, False if stopped by the RAM guard.
    """
    folder_path = KNOWN_FOLDERS[folder_alias]
    collection_name, current, removed, to_index = _plan_sync(folder_path)

    if collection_name not in _state:
        # First time for this folder/config combo: start from a clean
        # collection, with no leftovers from any previously interrupted run.
        try:
            _chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass
        _state[collection_name] = {}
    known = _state[collection_name]

    collection = _chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=_embedding_model
    )

    for path in removed:
        collection.delete(where={"path": path})
        known.pop(path, None)
    if removed:
        _save_state()

    if not to_index:
        logger.info("Index already up to date.")
        return True

    skipped = []
    list(_iter_candidate_files(folder_path, skipped))
    if skipped:
        logger.info(f"Skipped (too large): {', '.join(skipped)}")

    logger.info(f"{len(to_index)} files to index in '{folder_alias}' ({len(removed)} removed).")
    start_rss = _rss_gb()

    for n, path in enumerate(to_index, start=1):
        filename = os.path.basename(path)
        logger.info(f"({n}/{len(to_index)}) {filename}")

        collection.delete(where={"path": path})   # clear any old chunks for this file
        chunk_count = _index_file(collection, path)
        if chunk_count == 0:
            logger.info("  no extractable text (scanned PDF?)")

        # Marked as done even if empty, so it's not retried on every run
        known[path] = current[path]
        _save_state()
        gc.collect()

        rss = _rss_gb()
        if rss is not None:
            logger.info(f"  process RAM: {rss:.1f} GB")
            if start_rss is not None and (rss - start_rss) > MAX_INDEX_RAM_GROWTH_GB:
                logger.warning(
                    "RAM usage too high, stopping. Already-indexed files remain "
                    "saved: rerun to continue."
                )
                return False

    logger.info("Indexing completed.")
    return True


def _auto_sync(folder_alias: str, folder_path: str) -> str | None:
    """Used during a search: updates the index only for small changes.
    Returns a notice string if something needs attention, else None."""
    now = time.time()
    last = _last_checked.get(folder_path)
    if last is not None and (now - last) < MTIME_CHECK_COOLDOWN_SECONDS:
        return _last_notice.get(folder_path)
    _last_checked[folder_path] = now

    _, _, removed, to_index = _plan_sync(folder_path)
    notice = None

    if removed or to_index:
        if len(to_index) > AUTO_INDEX_MAX_FILES:
            notice = (f"Ci sono {len(to_index)} file non ancora indicizzati in questa cartella. "
                      f"Chiudi l'assistente ed esegui: python index_folders.py \"{folder_alias}\"")
        elif not sync_folder(folder_alias):
            notice = "L'indicizzazione si è interrotta per l'uso eccessivo di memoria."

    _last_notice[folder_path] = notice
    return notice


# ---------------------------- Search ----------------------------

def search_in_folder(folder_alias: str, query: str) -> str:
    """Searches the files of a known folder, keeping the index up to date
    when that's cheap enough to do inline."""
    folder_path = KNOWN_FOLDERS.get(folder_alias)
    if not folder_path or not os.path.isdir(folder_path):
        return f"Cartella non trovata. Cartelle disponibili: {', '.join(KNOWN_FOLDERS.keys())}."

    notice = _auto_sync(folder_alias, folder_path)

    collection = _chroma_client.get_or_create_collection(
        name=_collection_name_for(folder_path), embedding_function=_embedding_model
    )
    if collection.count() == 0:
        return notice or "Questa cartella non contiene testo indicizzato."

    results = collection.query(query_texts=[_clean_query(query, folder_alias)], n_results=N_RESULTS)
    if not results["documents"][0]:
        return "Nessun risultato trovato in quella cartella."

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        location = meta["source"]
        if "page" in meta:
            location += f" (pagina {meta['page']})"
        output.append(f"Nel file {location}: {doc.strip()}")

    body = "\n---\n".join(output)
    return f"[Nota: {notice}]\n{body}" if notice else body
