# Relacionadas con el entrenamiento
GENERACIONES                = 400
THREADS                     = 10
RENDER_MODE                 = None
CARGAR_CHECKPOINT           = True

FRAME_SKIP                  = 4
CANTIDAD_MAPAS_A_ENTRENAR   = 1 # No cambiar, solo hay 1 mapa configurado

# Relacionadas con su registro
CARPETA_CHECKPOINTS         = "generaciones_v7"
NOMBRE_MEJOR_AGENTE         = "mejor_agente"

# Otras constantes

BOTONES_USADOS              = ['A', 'B', 'C', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']

# ================================
# Definiciones de MOVIMIENTOS y COMBOS
# ================================

MOVIMIENTOS_BASE = {
    "q": [0,0,0,0,0,0,0,0],  # Quieto
    "s": [0,0,0,0,1,0,0,0],  # Saltar
    "a": [0,0,0,0,0,1,0,0],  # Agacharse
    "i": [0,0,0,0,0,0,1,0],  # Caminar izquierda
    "d": [0,0,0,0,0,0,0,1],  # Caminar derecha
    "si": [0,0,0,0,1,0,1,0], # Saltar izquierda
    "sd": [0,0,0,0,1,0,0,1], # Saltar derecha
    "b": [0,0,0,1,0,0,0,0],  # Bloquear
    "ba": [0,0,0,1,0,1,0,0], # Bloquear agachado
    "pa": [0,0,1,0,0,0,0,0], # Patada alta
    "pn": [0,1,0,0,0,0,0,0], # Patada normal
    "gn": [1,0,0,0,0,0,0,0], # Golpe normal
}

COMBOS_JAX = [
    ["d", "d", "gn"],             # Agarre de fuerza
    ["d", "a", "i", "pa"],        # Onda de energía
    ["a", "gn"]                   # Golpe martillo
]
