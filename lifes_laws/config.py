# Relacionadas con el entrenamiento
GENERACIONES                = 100
THREADS                     = 3
RENDER_MODE                 = None
CARGAR_CHECKPOINT           = True

FRAME_SKIP                  = 4
CANTIDAD_MAPAS_A_ENTRENAR   = 1 # No cambiar, solo hay 1 mapa configurado

# Relacionadas con su registro
CARPETA_CHECKPOINTS         = "generaciones_v2"
NOMBRE_MEJOR_AGENTE         = "mejor_agente"

# Otras constantes

BOTONES_USADOS              = ['A', 'B', 'C', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']
MOVIMIENTOS_BASE = {
    (0, 0, 0, 0, 1, 0, 0, 0): "s",   # Saltar
    (0, 0, 0, 0, 0, 1, 0, 0): "a",   # Agacharse
    (0, 0, 0, 0, 0, 0, 1, 0): "i",   # Caminar izquierda
    (0, 0, 0, 0, 0, 0, 0, 1): "d",   # Caminar derecha
    (0, 0, 0, 0, 1, 0, 1, 0): "si",  # Saltar izquierda
    (0, 0, 0, 0, 1, 0, 0, 1): "sd",  # Saltar derecha
    (0, 0, 0, 1, 0, 0, 0, 0): "b",   # Bloquear
    (0, 0, 0, 1, 0, 1, 0, 0): "ba",  # Bloquear agachado
    (0, 0, 1, 0, 0, 0, 0, 0): "pa",  # Patada alta
    (0, 1, 0, 0, 0, 0, 0, 0): "pn",  # Patada normal
    (1, 0, 0, 0, 0, 0, 0, 0): "gn",  # Golpe normal
}


COMBOS_LIUKANG              = [
    ["s", "d", "d", "gn"], # Bola de fuego en el aire
    ["d", "d", "pa"],  # Patada voladora
    ["d", "d", "gn"], # Bola de fuego
]

COMBOS_JAX                  = [
    ["d", "d", "gn"], # Agarre de fuerza
    ["d", "a", "i", "pa"], # Onda de energia
    ["a", "gn"]                 # Golpe martillo
]