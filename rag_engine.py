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
    _process = None  # senza psutil funziona lo stesso, ma senza monitoraggio della RAM

# pypdf emette un warning per ogni pagina con certi font: non sono errori.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ============================ CONFIGURAZIONE ============================

# Alias parlato -> percorso reale. RIMETTI QUI I TUOI ALIAS E PERCORSI.
KNOWN_FOLDERS = {
    "test": r"D:\Francy\Documenti\Prova"
}

# Parole che, insieme a un alias riconosciuto, attivano la ricerca nei file
SEARCH_TRIGGER_WORDS = {"cerca", "trova", "cercami", "trovami", "cartella", "file"}

SUPPORTED_EXTENSIONS = (".py", ".md", ".txt", ".pdf")

# Cartelle mai scandagliate. In più vengono saltate TUTTE le cartelle nascoste
# (che iniziano con ".") e qualsiasi virtualenv, riconosciuto dal file pyvenv.cfg.
EXCLUDED_DIRS = {
    "venv", ".venv", "env", "site-packages", "dist-packages", "node_modules",
    "__pycache__", "chroma_db", "folder_cache_db", "build", "dist",
}

MAX_FILE_SIZE_MB = 50        # file più grandi vengono saltati
MAX_CHUNKS_PER_FILE = 6000   # tetto di sicurezza per singolo file

# Il modello multilingue tronca il testo a 128 token (~500 caratteri): chunk più lunghi
# verrebbero "visti" solo in parte dall'embedding. Con 500/100 il testo è coperto tutto.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

UPSERT_BATCH_SIZE = 128      # chunk embeddati e salvati per volta (tiene bassa la RAM)
N_RESULTS = 4                # chunk restituiti per ricerca

# Guardia di sicurezza: se durante l'indicizzazione la RAM del processo cresce di più
# di questo valore, si ferma (i file già fatti restano salvati) invece di bloccare il PC.
MAX_INDEX_RAM_GROWTH_GB = 6.0

# Dalla voce si indicizzano al volo al massimo questi file nuovi/modificati;
# oltre, si chiede di lanciare lo script index_folders.py a assistente spento.
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
    # Scrittura atomica: se il PC si blocca a metà, il file di stato non si corrompe
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_state, f)
    os.replace(tmp_path, STATE_FILE)


# collection_name -> {percorso_file: firma}. Ricorda, file per file, cosa è già indicizzato.
_state = _load_state()
_last_checked = {}   # folder_path -> timestamp dell'ultimo controllo (solo in memoria)
_last_notice = {}    # folder_path -> ultimo avviso generato


def _rss_gb() -> float | None:
    return _process.memory_info().rss / (1024 ** 3) if _process else None


def _collection_name_for(folder_path: str) -> str:
    # Modello e dimensione dei chunk fanno parte della chiave: se cambiano, si crea
    # un indice nuovo invece di mischiare embedding non confrontabili.
    key = f"{folder_path}|{EMBEDDING_MODEL_NAME}|{CHUNK_SIZE}|{CHUNK_OVERLAP}"
    return "folder_" + hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------- Riconoscimento cartella ----------------------------

def resolve_folder_alias(spoken_text: str, cutoff: float = 0.6) -> str | None:
    """Cerca un alias noto ALL'INTERNO di una frase (esatto, poi fuzzy per tollerare
    gli errori di trascrizione di Whisper). Ritorna l'ALIAS, non il percorso."""
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
    """(True, alias) se la frase contiene sia un alias riconosciuto sia una parola-trigger."""
    spoken_lower = spoken_text.lower()
    alias = resolve_folder_alias(spoken_lower)
    if alias is None:
        return False, None

    has_trigger = any(word in spoken_lower for word in SEARCH_TRIGGER_WORDS)
    return (True, alias) if has_trigger else (False, None)


def _clean_query(query: str, folder_alias: str) -> str:
    """Toglie alias e parole di comando: resta solo cosa cercare."""
    cleaned = query.lower().replace(folder_alias, " ")
    words = [w for w in cleaned.split() if w.strip(",.?!") not in SEARCH_TRIGGER_WORDS]
    return " ".join(words).strip() or query


# ---------------------------- Scansione file ----------------------------

def _iter_candidate_files(folder_path: str, skipped: list | None = None):
    for root, dirs, files in os.walk(folder_path):
        if "pyvenv.cfg" in files:      # è un virtualenv, con qualunque nome
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
    """Genera (chunk, numero_pagina) uno alla volta: la RAM per file resta limitata.
    Il numero pagina è None per i file non-PDF."""
    if filepath.lower().endswith(".pdf"):
        try:
            reader = PdfReader(filepath)
            total_pages = len(reader.pages)
        except Exception:
            return
        for page_number, page in enumerate(reader.pages, start=1):
            if page_number % 50 == 0:
                print(f"[INDEX]   ...pagina {page_number}/{total_pages}")
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


# ---------------------------- Indicizzazione ----------------------------

def _index_file(collection, filepath: str) -> int:
    """Indicizza un singolo file a piccoli blocchi. Ritorna il numero di chunk salvati."""
    filename = os.path.basename(filepath)
    path_hash = hashlib.md5(filepath.encode()).hexdigest()[:16]

    docs, metadatas, ids = [], [], []
    total = 0

    for i, (chunk, page) in enumerate(_iter_file_chunks(filepath)):
        if i >= MAX_CHUNKS_PER_FILE:
            print(f"[INDEX]   limite di {MAX_CHUNKS_PER_FILE} chunk raggiunto: il resto del file viene ignorato")
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
    """Confronta i file su disco con quelli già indicizzati.
    Ritorna (collection_name, current, removed, to_index)."""
    collection_name = _collection_name_for(folder_path)
    known = _state.get(collection_name, {})

    current = {}
    for path in _iter_candidate_files(folder_path):
        signature = _file_signature(path)
        if signature:
            current[path] = signature

    removed = [p for p in known if p not in current]
    to_index = [p for p, sig in current.items() if known.get(p) != sig]
    to_index.sort(key=_file_size)   # prima i piccoli: risultati rapidi, i grossi per ultimi
    return collection_name, current, removed, to_index


def sync_folder(folder_alias: str) -> bool:
    """
    Allinea l'indice alla cartella: indicizza i file nuovi/modificati, toglie quelli
    cancellati. Si salva file per file, quindi se viene interrotta riparte da dove era.
    Ritorna True se ha completato, False se si è fermata per la guardia sulla RAM.
    """
    folder_path = KNOWN_FOLDERS[folder_alias]
    collection_name, current, removed, to_index = _plan_sync(folder_path)

    if collection_name not in _state:
        # Prima volta per questa cartella/configurazione: si parte da una collection
        # pulita, senza residui di eventuali run interrotte in passato.
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
        print("[INDEX] Indice già aggiornato.")
        return True

    skipped = []
    list(_iter_candidate_files(folder_path, skipped))
    if skipped:
        print(f"[INDEX] Saltati perché troppo grandi: {', '.join(skipped)}")

    print(f"[INDEX] {len(to_index)} file da indicizzare in '{folder_alias}' ({len(removed)} rimossi).")
    start_rss = _rss_gb()

    for n, path in enumerate(to_index, start=1):
        filename = os.path.basename(path)
        print(f"[INDEX] ({n}/{len(to_index)}) {filename}")

        collection.delete(where={"path": path})   # via eventuali vecchi chunk di questo file
        chunk_count = _index_file(collection, path)
        if chunk_count == 0:
            print("[INDEX]   nessun testo estraibile (PDF scansionato?)")

        # Segnato come fatto anche se vuoto, per non riprovarlo ad ogni avvio
        known[path] = current[path]
        _save_state()
        gc.collect()

        rss = _rss_gb()
        if rss is not None:
            print(f"[INDEX]   RAM del processo: {rss:.1f} GB")
            if start_rss is not None and (rss - start_rss) > MAX_INDEX_RAM_GROWTH_GB:
                print("[INDEX] ATTENZIONE: consumo di RAM troppo alto, mi fermo. "
                      "I file già indicizzati restano salvati: rilancia per continuare.")
                return False

    print("[INDEX] Indicizzazione completata.")
    return True


def _auto_sync(folder_alias: str, folder_path: str) -> str | None:
    """Usata durante una ricerca: aggiorna l'indice solo per piccole modifiche.
    Ritorna un avviso testuale se qualcosa richiede attenzione, altrimenti None."""
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


# ---------------------------- Ricerca ----------------------------

def search_in_folder(folder_alias: str, query: str) -> str:
    """Cerca nei file di una cartella nota, tenendo l'indice aggiornato quando è poco lavoro."""
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
    if not results['documents'][0]:
        return "Nessun risultato trovato in quella cartella."

    output = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        location = meta['source']
        if 'page' in meta:
            location += f" (pagina {meta['page']})"
        output.append(f"Nel file {location}: {doc.strip()}")

    body = "\n---\n".join(output)
    return f"[Nota: {notice}]\n{body}" if notice else body
