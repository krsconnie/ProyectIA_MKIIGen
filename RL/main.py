import retro
import numpy as np

# Cargar el entorno con dos jugadores
env = retro.make(game='MortalKombatII-Genesis', players=1, use_restricted_actions=retro.Actions.ALL)
obs = env.reset()

# Variables para seguir la salud anterior
p1_health_anterior = 120
p2_health_anterior = 120


# Loop de simulación
for _ in range(5000):
    env.render()

    # Acciones aleatorias para ambos jugadores
    action_p1 = env.action_space.sample()

    # Paso en el entorno
    obs, reward, terminated, truncated, info = env.step(action_p1)
    done = terminated or truncated

    # Extraer salud actual
    p1_health_actual = info['health']
    p2_health_actual = info['enemy_health']

    # Calcular recompensa individual como daño hecho
    reward_p1 = p2_health_anterior - p2_health_actual

    # Mostrar la info
    print(f"P1 ➡️ Recompensa: {reward_p1} | Salud: {p1_health_actual}")
    print("-" * 40)

    # Actualizar salud anterior
    p1_health_anterior = p1_health_actual
    p2_health_anterior = p2_health_actual

    if done:
        break

env.close()
