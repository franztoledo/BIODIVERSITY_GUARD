"""Genera el zip de entrega (y, opcional, la carpeta minima de despliegue).

Uso:
    python empaquetar.py                         -> TB2_1ACC0057_16623_GRUPO_3.zip
    python empaquetar.py MI_NOMBRE.zip           -> zip con otro nombre
    python empaquetar.py --despliegue CARPETA    -> copia solo los archivos minimos para publicar el dashboard (ver DESPLIEGUE.md)

Estructura del zip:
    /codigo/frontend  /codigo/backend (sin .db)  /codigo/ia_models  /codigo/notebooks
    /datos/datasets_preparation  /datos/modulo_ia_simulado  (/datos/datasets_eda si el zip total no pasa de 150 MB)
    /docs  /resultados/*.csv  README.md  requirements.txt  .streamlit/config.toml
"""
import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
NOMBRE_POR_DEFECTO = "TB2_1ACC0057_16623_GRUPO_3.zip"
LIMITE_MB = 150
EXCLUIR_PARTES = {".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".venv", "venv", "node_modules"}
EXCLUIR_SUFIJOS = {".db", ".creando", ".pyc", ".pyo", ".db-journal", ".db-wal", ".db-shm"}
EXCLUIR_NOMBRES = {".DS_Store", "Thumbs.db"}

# Archivos minimos para correr el servicio real sin los datasets pesados (ver DESPLIEGUE.md)
MINIMOS = ["requirements.txt", ".streamlit/config.toml", "frontend", "backend", "ia_models",
           "datos/datasets_preparation/kpi_parametros.json", "datos/datasets_preparation/parametros_preparacion.json",
           "datos/datasets_eda/dataset_panel_semanal_COMPLETO.csv", "datos/modulo_ia_simulado", "docs/fichas_kpi.json",
           "backend/semilla/biodiversity_guard_semilla.db"]  # base semilla (hasta 2026-08-31): arranque rapido en la nube


def excluido(ruta: Path) -> bool:
    partes = set(ruta.relative_to(RAIZ).parts) if ruta.is_relative_to(RAIZ) else set(ruta.parts)
    if partes & EXCLUIR_PARTES or ruta.name in EXCLUIR_NOMBRES or ruta.suffix in EXCLUIR_SUFIJOS:
        return True
    return any(p.startswith("cache_tuning") for p in ruta.parts)


def archivos(carpeta: Path, patron="*", recursivo=True):
    it = carpeta.rglob(patron) if recursivo else carpeta.glob(patron)
    return sorted(p for p in it if p.is_file() and not excluido(p))


def plan_base() -> list[tuple[Path, str]]:
    """(origen, ruta dentro del zip) de todo menos datasets_eda."""
    p: list[tuple[Path, str]] = []
    for sub in ("frontend", "backend", "ia_models"):
        p += [(f, f"codigo/{f.relative_to(RAIZ).as_posix()}") for f in archivos(RAIZ / sub)]
    # notebooks: todos los .ipynb de codigo/, conservando la subcarpeta (preprocesamiento, EDA, Data Preparation, Modelado...)
    for f in archivos(RAIZ / "codigo", "*.ipynb"):
        p.append((f, f"codigo/notebooks/{f.relative_to(RAIZ / 'codigo').as_posix()}"))
    for sub in ("datasets_preparation", "modulo_ia_simulado"):
        p += [(f, f"datos/{sub}/{f.relative_to(RAIZ / 'datos' / sub).as_posix()}") for f in archivos(RAIZ / "datos" / sub)]
    p += [(f, f"docs/{f.relative_to(RAIZ / 'docs').as_posix()}") for f in archivos(RAIZ / "docs")]
    p += [(f, f"resultados/{f.name}") for f in archivos(RAIZ / "resultados", "*.csv", recursivo=False)]
    for extra in ("README.md", "requirements.txt", ".streamlit/config.toml", "empaquetar.py"):
        if (RAIZ / extra).exists():
            p.append((RAIZ / extra, extra))
    return p


def escribir(zf: zipfile.ZipFile, items):
    for origen, destino in items:
        zf.write(origen, destino)


def empaquetar(nombre: str) -> Path:
    destino = Path(nombre)
    if not destino.is_absolute():
        destino = RAIZ / destino
    if destino.exists():
        destino.unlink()
    base = plan_base()
    eda = [(f, f"datos/datasets_eda/{f.relative_to(RAIZ / 'datos' / 'datasets_eda').as_posix()}") for f in archivos(RAIZ / "datos" / "datasets_eda")]
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        escribir(zf, base)
    tam_base = destino.stat().st_size
    # datasets_eda solo si el total comprimido no supera el limite
    with tempfile.TemporaryDirectory() as t:
        prueba = Path(t) / "eda.zip"
        with zipfile.ZipFile(prueba, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            escribir(zf, eda)
        tam_eda = prueba.stat().st_size
    incluye = (tam_base + tam_eda) / 1e6 <= LIMITE_MB
    print(f"Zip sin datasets_eda: {tam_base / 1e6:,.1f} MB | datasets_eda comprimido: {tam_eda / 1e6:,.1f} MB | limite {LIMITE_MB} MB -> "
          f"{'se incluye' if incluye else 'NO se incluye'} datasets_eda")
    if incluye:
        with zipfile.ZipFile(destino, "a", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            escribir(zf, eda)
    return destino


def imprimir_arbol(zip_path: Path, profundidad=3):
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
    carpetas: dict[str, list[int]] = {}
    for i in infos:
        partes = i.filename.split("/")
        for n in range(1, min(len(partes), profundidad) + 1):
            clave = "/".join(partes[:n]) + ("/" if n < len(partes) else "")
            v = carpetas.setdefault(clave, [0, 0, 0])
            v[0] += 1
            v[1] += i.file_size
            v[2] += i.compress_size
    print(f"\nArbol del zip (hasta {profundidad} niveles; archivos, tamano sin comprimir, comprimido):")
    for k in sorted(carpetas):
        n, s, c = carpetas[k]
        nivel = k.rstrip("/").count("/")
        etiqueta = k.rstrip("/").split("/")[-1] + ("/" if k.endswith("/") else "")
        print(f"{'  ' * nivel}{etiqueta:<45s} {n:5d} archivos  {s / 1e6:8.1f} MB  {c / 1e6:8.1f} MB" if k.endswith("/") or n == 1 else "")
    print(f"\nTotal: {len(infos)} archivos | {sum(i.file_size for i in infos) / 1e6:,.1f} MB sin comprimir | zip {zip_path.stat().st_size / 1e6:,.1f} MB ({zip_path})")
    mayor = max(infos, key=lambda i: i.file_size)
    print(f"Archivo mas grande: {mayor.filename} ({mayor.file_size / 1e6:,.1f} MB)")


def despliegue(carpeta: str):
    """Copia solo los archivos minimos para publicar el dashboard (repositorio de despliegue)."""
    dest = Path(carpeta).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for rel in MINIMOS:
        o = RAIZ / rel
        fs = [o] if o.is_file() else archivos(o)  # el .db de la semilla se pide por su ruta exacta (es archivo, no pasa por el filtro)
        for f in fs:
            d = dest / f.relative_to(RAIZ)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d)
            total += f.stat().st_size
    print(f"Copiados los archivos minimos en {dest}: {total / 1e6:,.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nombre", nargs="?", default=NOMBRE_POR_DEFECTO, help=f"nombre del zip (por defecto {NOMBRE_POR_DEFECTO})")
    ap.add_argument("--despliegue", metavar="CARPETA", help="solo copia los archivos minimos de despliegue a CARPETA")
    a = ap.parse_args()
    if a.despliegue:
        despliegue(a.despliegue)
        sys.exit(0)
    z = empaquetar(a.nombre)
    imprimir_arbol(z)
