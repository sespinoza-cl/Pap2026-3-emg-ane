"""Configuracion central del proyecto Paper3 - EMG masticatorio bajo anestesia."""
from pathlib import Path

# --- Rutas ---
RAW_DIR    = Path(r"D:\Exp1\Exp1\Raw data\Raw\All_36")
PROJ       = Path(r"C:\Users\Pc - Casa\Desktop\Proyectos_Claude\Phd\Paper3")
DERIV      = PROJ / "data_derived"
OUT        = PROJ / "outputs"
for d in (DERIV, OUT):
    d.mkdir(parents=True, exist_ok=True)

# --- Adquisicion ---
FS_ORIG    = 1024.0          # Hz (Biosemi)
# EMG = canales 1-idx 69:72 = EXG5..EXG8 -> 0-idx 68:72
EMG_LABELS = ["EXG5", "EXG6", "EXG7", "EXG8"]

# --- Filtrado EMG (en fs original, anti-aliasing: filtrar ANTES de cualquier resample) ---
BP_LOW     = 20.0            # Hz
BP_HIGH    = 450.0           # Hz
BP_ORDER   = 4              # Butterworth (filtfilt -> orden efectivo x2, fase cero)
# Ruido de linea: se elimina con Zapline-plus (no notch) para preservar el
# espectro; ver 10_preprocess.LINE_NOISEFREQS.
# Se mantiene 1024 Hz (sin downsample) para preservar la banda EMG hasta 450 Hz
# y una estimacion fiel de la frecuencia mediana.

# --- Segmentacion ---
CHEW_CODE_ON   = 1          # inicio de bout de masticacion
CHEW_CODE_OFF  = 2          # fin de bout
RS_CODES       = [3, 4, 5, 6]
BOUT_DUR_S     = 60.0       # duracion nominal de cada bout
N_BOUTS_TOTAL  = 8
RS_WIN         = (5.0, 95.0)  # ventana usada por bloque de reposo (s desde el marcador)

# --- Mapeo condicion (A1=Anestesia, A2=Placebo; confirmado por el usuario) ---
# R1: bouts 1-4 = Anestesia (A1), 5-8 = Placebo (A2)
# R2: bouts 1-4 = Placebo  (A2), 5-8 = Anestesia (A1)  [counterbalanceo]
def bout_condition_map(rama):
    """Devuelve dict {bout_index(0-7): 'ANE'|'PLA'} segun rama '1' o '2'."""
    if str(rama) == "1":
        first, second = "ANE", "PLA"
    else:
        first, second = "PLA", "ANE"
    return {**{i: first for i in range(0, 4)}, **{i: second for i in range(4, 8)}}

# --- Sujetos (los 36; inclusion final por QC propio) ---
SUBJECTS_36 = ["M1","M2","M3","M4","M5","M6","M7",
               "PS1","PS2","PS3","PS4","PS5","PS6","PS7","PS8","PS9",
               "PS10","PS11","PS12","PS13","PS14","PS15",
               "S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13","S14"]
# Listas previas del laboratorio (referencia para sensibilidad)
LISTA30 = ["M1","M4","M5","M6","PS10","PS11","PS13","PS14","PS15","PS1","PS2","PS3","PS4",
           "PS5","PS6","PS7","PS8","PS9","S10","S11","S13","S14","S1","S2","S4","S5","S6","S7","S8","S9"]
LISTA32 = LISTA30 + ["M2","S3"]
