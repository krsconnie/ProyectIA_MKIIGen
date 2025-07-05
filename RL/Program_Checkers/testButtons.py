import sys
sys.path.append('./RL')
import retro
import time

env = retro.make(game='MortalKombatII-Genesis', players=2)
obs = env.reset()

# Esperar a que empiece la pelea (ajusta el número si hace falta)
for _ in range(180):
    obs, _, terminated, truncated, info= env.step([0]*24)
    env.render()
# ------------------------------------
#print(env.buttons)
BOTONES_USADOS = ['A', 'B', 'C', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']
INDICES_BOTONES = [env.buttons.index(b) for b in BOTONES_USADOS]
print(INDICES_BOTONES)

# attack buttons = B, A, C
attack_buttons = [0, 1, 8]
# block buttons = START
block_buttons = [3]
# Movement_buttons = UP, DOWN, LEFT, RIGHT
movement_buttons = [4, 5, 6, 7]
# ------------------------------------
# Crear vector de acción solo con ese botón presionado
accion1 = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
accion2 = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
# ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

print(f"Ejecutando acción índice {accion1}")

# Mantener la acción para que se vea la animación (15 frames)
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(accion1)
    env.render()
    """"
    if any(accion1[i] for i in attack_buttons):
        print("Acción de Ataque")
    elif any(accion1[i] for i in block_buttons):
        print("Acción de Defenza")
    """
    #print(info)
    #obs, reward, terminated, truncated, info = env.step(accion2)
    #env.render()
    #time.sleep(0.03)

print(f"Ejecutando acción índice {accion2}")
for _ in range(180):
    obs, reward, terminated, truncated, info = env.step(accion2)
    env.render()
    """
    if any(accion2[i] for i in attack_buttons):
        print("Acción de Ataque")
    elif any(accion2[i] for i in block_buttons):
        print("Acción de Defenza")
    """
    #print(info)
for _ in range(70):
    #print(info)
    obs, _, terminated, truncated, _= env.step([0]*24)
    env.render()

env.close()
