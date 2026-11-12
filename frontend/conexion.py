"""Selecciona el servicio: real por defecto; stub si BG_SERVICIO=stub o si el real falla."""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def obtener_servicio():
    """Devuelve (modulo, aviso). aviso es None si todo va bien."""
    aviso = None
    if os.environ.get("BG_SERVICIO", "real").lower() != "stub":
        if str(RAIZ) not in sys.path:
            sys.path.insert(0, str(RAIZ))
        try:
            from backend import servicio  # type: ignore
            return servicio, None
        except Exception as e:  # noqa: BLE001
            aviso = f"No se pudo cargar backend.servicio ({type(e).__name__}). Se usa el servicio de ejemplo."
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import servicio_stub
    return servicio_stub, aviso


def semana_ejemplo(sv):
    """Próximo CSV de datos/modulo_ia_simulado/ (el que sigue a la última semana). Devuelve (nombre, bytes) o None."""
    if hasattr(sv, "semana_ejemplo"):
        return sv.semana_ejemplo()
    from datetime import date, timedelta
    sig = (date.fromisoformat(sv.ultima_semana()) + timedelta(days=7)).isoformat()
    f = RAIZ / "datos" / "modulo_ia_simulado" / f"semana_{sig}.csv"
    return (f.name, f.read_bytes()) if f.exists() else None
