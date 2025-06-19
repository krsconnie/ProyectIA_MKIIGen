import retro
import time

env = retro.make(game='MortalKombatII-Genesis', players=2)
obs = env.reset()

# Esperar a que empiece la pelea (ajusta el número si hace falta)
for _ in range(180):
    obs, _, terminated, truncated, _= env.step([0]*24)
    env.render()
# ------------------------------------
# Cambia aquí el índice de la acción que querés probar (0 a 11)
print(env.buttons)

# ------------------------------------
# Crear vector de acción solo con ese botón presionado
accion1 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
accion2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

print(f"Ejecutando acción índice {accion1}")

# Mantener la acción para que se vea la animación (15 frames)
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(accion1)
    env.render()
    print(info)
    #obs, reward, terminated, truncated, info = env.step(accion2)
    #env.render()
    #time.sleep(0.03)
print("Sgunda accion")
for _ in range(180):
    obs, reward, terminated, truncated, info = env.step(accion2)
    env.render()
    print(info)
for _ in range(70):
    print(info)
    obs, _, terminated, truncated, _= env.step([0]*24)
    env.render()


print("Acción terminada. Cambiá 'accion_a_probar' para probar otra.")

env.close()
