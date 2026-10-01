"""
Indicizza le cartelle di KNOWN_FOLDERS FUORI dall'assistente vocale.

Uso (con l'assistente vocale CHIUSO, così RAM e VRAM sono tutte libere):
    python index_folders.py                    -> tutte le cartelle
    python index_folders.py università         -> solo quell'alias
    python index_folders.py "progetto codice"  -> alias con spazi tra virgolette

Si può interrompere con Ctrl+C e rilanciare: riparte dal punto in cui era.
"""
import sys
from rag_engine import KNOWN_FOLDERS, sync_folder


def main():
    aliases = sys.argv[1:] or list(KNOWN_FOLDERS.keys())

    for alias in aliases:
        if alias not in KNOWN_FOLDERS:
            print(f"Alias sconosciuto: '{alias}'. Disponibili: {', '.join(KNOWN_FOLDERS)}")
            continue
        print(f"\n=== Indicizzazione: {alias} ===")
        try:
            sync_folder(alias)
        except KeyboardInterrupt:
            print("\n[INDEX] Interrotto. I file già indicizzati restano salvati: rilancia per continuare.")
            break


if __name__ == "__main__":
    main()
