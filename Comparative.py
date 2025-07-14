import retro
import pickle
import neat
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

# ===========
# Configuración general
# ===========
GAME = 'MortalKombatII-Genesis'
STATE = "Level1.JaxVsBaraka"
MAX_FRAMES = 4500
CONFIG_PATH = "config-neat"
GENOME_PKL = "homunculo.pkl"

# ===========
# Parámetro de ruido
# ===========
NOISE_PROB = 0.1  # 10% de ruido en las acciones del agente NEAT

# ===========
# Preprocesado de imagen
# ===========
def preprocess(obs):
    obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs_resized = cv2.resize(obs_gray, (84, 84))
    return obs_resized.flatten() / 255.0

# ===========
# Avanzar hasta detectar pelea
# ===========
def avanzar_hasta_pelea(env, obs, info):
    for _ in range(500):
        timer = info.get("timer", 99)
        player_health = info.get("health", 120)
        enemy_health = info.get("enemy_health", 120)
        if timer < 99 or player_health < 120 or enemy_health < 120:
            print("✅ Pelea detectada.")
            return obs, info
        obs, _, terminated, truncated, info = env.step([0]*len(env.buttons))
        if terminated or truncated:
            obs, info = env.reset()
    print("⚠️ Advertencia: no se detectó pelea tras avanzar frames.")
    return obs, info

# ===========
# Evaluador del agente NEAT
# ===========
def evaluar_agente_NEAT(config, genome, episodios=7, repeticiones=10, noise_prob=NOISE_PROB):
    todas_recompensas = []

    for rep in range(repeticiones):
        recompensas_totales = []

        for ep in range(episodios):
            print(f"▶️ Mostrando combate NEAT rep {rep+1}, episodio {ep+1}...")
            env = retro.make(game=GAME, state=STATE, render_mode="human")
            obs, info = env.reset()

            obs, info = avanzar_hasta_pelea(env, obs, info)
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            done = False
            frame = 0
            recompensa_ep = 0

            max_x = 320.0
            max_y = 224.0

            last_enemy_health = info.get("enemy_health", 120)
            last_player_health = info.get("health", 120)

            ventana_frames = 120
            frame_window_counter = 0
            damage_acumulado_en_ventana = 0

            while not done and frame < MAX_FRAMES:
                pos_data = [
                    info.get("x_position", 0) / max_x,
                    info.get("y_position", 0) / max_y,
                    info.get("enemy_x_position", 0) / max_x,
                    info.get("enemy_y_position", 0) / max_y
                ]
                obs_proc = preprocess(obs)
                inputs = np.concatenate([obs_proc, pos_data])

                output = net.activate(inputs)
                action = [1 if o > 0.5 else 0 for o in output]

                # ✅ Introducir ruido con probabilidad noise_prob
                if random.random() < noise_prob:
                    action = [random.randint(0, 1) for _ in env.buttons]

                obs, _, terminated, truncated, info = env.step(action)
                frame += 1

                if info.get("rounds_won", 0) > 0:
                    print("✅ Jugador NEAT ganó el round!")
                    done = True
                elif info.get("enemy_rounds_won", 0) > 0:
                    print("❌ Enemigo ganó el round contra NEAT!")
                    done = True
                elif terminated or truncated:
                    done = True

                if any(action):
                    recompensa_ep += 2

                enemy_health = info.get("enemy_health", last_enemy_health)
                player_health = info.get("health", 120)

                enemy_damage = max(0, last_enemy_health - enemy_health)
                self_damage = max(0, last_player_health - player_health)

                if enemy_damage > 0:
                    recompensa_ep += enemy_damage * 2.0
                if self_damage > 0:
                    recompensa_ep -= self_damage

                frame_window_counter += 1
                damage_acumulado_en_ventana += enemy_damage

                if frame_window_counter >= ventana_frames:
                    if damage_acumulado_en_ventana > 0:
                        recompensa_ep += 200
                    else:
                        recompensa_ep -= 100
                    frame_window_counter = 0
                    damage_acumulado_en_ventana = 0

                last_enemy_health = enemy_health
                last_player_health = player_health

            env.close()
            recompensas_totales.append(recompensa_ep)

        todas_recompensas.append(recompensas_totales)
    return todas_recompensas

# ===========
# Evaluador Random
# ===========
def evaluar_random(episodios=7, repeticiones=10):
    todas_recompensas = []

    for rep in range(repeticiones):
        recompensas_totales = []

        for ep in range(episodios):
            print(f"▶️ Mostrando combate RANDOM rep {rep+1}, episodio {ep+1}...")
            env = retro.make(game=GAME, state=STATE, render_mode="human")
            obs, info = env.reset()

            obs, info = avanzar_hasta_pelea(env, obs, info)

            done = False
            frame = 0
            recompensa_ep = 0

            last_enemy_health = info.get("enemy_health", 120)
            last_player_health = info.get("health", 120)

            ventana_frames = 120
            frame_window_counter = 0
            damage_acumulado_en_ventana = 0

            while not done and frame < MAX_FRAMES:
                action = [random.randint(0, 1) for _ in env.buttons]
                obs, _, terminated, truncated, info = env.step(action)
                frame += 1

                if info.get("rounds_won", 0) > 0:
                    print("✅ Jugador RANDOM ganó el round!")
                    done = True
                elif info.get("enemy_rounds_won", 0) > 0:
                    print("❌ Enemigo ganó el round contra RANDOM!")
                    done = True
                elif terminated or truncated:
                    done = True

                if any(action):
                    recompensa_ep += 2

                enemy_health = info.get("enemy_health", last_enemy_health)
                player_health = info.get("health", 120)

                enemy_damage = max(0, last_enemy_health - enemy_health)
                self_damage = max(0, last_player_health - player_health)

                if enemy_damage > 0:
                    recompensa_ep += enemy_damage * 2.0
                if self_damage > 0:
                    recompensa_ep -= self_damage

                frame_window_counter += 1
                damage_acumulado_en_ventana += enemy_damage

                if frame_window_counter >= ventana_frames:
                    if damage_acumulado_en_ventana > 0:
                        recompensa_ep += 200
                    else:
                        recompensa_ep -= 100
                    frame_window_counter = 0
                    damage_acumulado_en_ventana = 0

                last_enemy_health = enemy_health
                last_player_health = player_health

            env.close()
            recompensas_totales.append(recompensa_ep)

        todas_recompensas.append(recompensas_totales)
    return todas_recompensas

# ===========
# Gráfico comparativo
# ===========
def graficar_comparacion(todas_recompensas_model, todas_recompensas_random):
    data_model = np.array(todas_recompensas_model)
    data_random = np.array(todas_recompensas_random)

    plt.figure(figsize=(6,5))
    posiciones = [1, 1.3]

    box = plt.boxplot(
        [data_model.flatten(), data_random.flatten()],
        patch_artist=True,
        positions=posiciones,
        widths=0.2,
        labels=["Agente NEAT", "Random"]
    )

    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    promedio_model = np.mean(data_model)
    promedio_random = np.mean(data_random)
    plt.scatter([posiciones[0]], [promedio_model], color='blue', label=f'Promedio NEAT: {promedio_model:.2f}', zorder=10)
    plt.scatter([posiciones[1]], [promedio_random], color='green', label=f'Promedio Random: {promedio_random:.2f}', zorder=10)

    plt.title("Recompensas totales: NEAT vs Random")
    plt.ylabel("Recompensa total por episodio")
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparacion_NEAT_vs_Random.png')
    plt.close()
    print("✅ Gráfico guardado: comparacion_NEAT_vs_Random.png")

# ===========
# MAIN
# ===========
if __name__ == "__main__":
    print("⚙️  Cargando configuración y genoma...")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )
    with open(GENOME_PKL, "rb") as f:
        genome = pickle.load(f)

    episodios = 7
    repeticiones = 10

    print("🏁 Evaluando agente NEAT...")
    todas_recompensas_model = evaluar_agente_NEAT(config, genome, episodios, repeticiones, noise_prob=NOISE_PROB)

    print("🏁 Evaluando agente aleatorio...")
    todas_recompensas_random = evaluar_random(episodios, repeticiones)

    print("📈 Generando gráfico comparativo...")
    graficar_comparacion(todas_recompensas_model, todas_recompensas_random)
