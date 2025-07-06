import lifes_laws.config as exe_config
from . import tools
import neat
import retro
import numpy as np
import random

"""
De lo más importante del NEAT, la evaluacion de los genomas.
"""
def eval_genome(genome, config):

    escenarios_disponibles = [
        "Level1.JaxVsBaraka",
        "Level1.JaxVsLiuKang"
    ]

    avg_fitness = [0] * len(escenarios_disponibles)

    for i_escn, escenario in enumerate(escenarios_disponibles):
        env = retro.make(game='MortalKombatII-Genesis', state=escenario, render_mode=exe_config.RENDER_MODE)
        obs, info = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        INDICES_BOTONES = [env.buttons.index(b) for b in exe_config.BOTONES_USADOS]
        action = [0] * len(env.buttons)

        # Inicializando variables.
        done = False
        frame_count = 0
        max_frames = 4500
        warmup_frames = 200
        last_enemy_health = 120
        last_player_health = 120

        # Variables nuevas para fitness complejo
        ventana_frames = 45
        frame_window_counter = 0
        damage_acumulado_en_ventana = 0
        penalizaciones_seguidas = 0
        umbral_penalizaciones = 7

        fitness = 0

        while not done and frame_count < max_frames:
            if frame_count < warmup_frames:
                action = env.action_space.sample()
                last_enemy_health = info.get("enemy_health", last_enemy_health)
                last_player_health = info.get("health", last_player_health)
            else:
                max_x_pos = 320.0
                max_y_pos = 224.0

                pos_data = [
                    info.get("x_position", 0) / max_x_pos,
                    info.get("y_position", 0) / max_y_pos,
                    info.get("enemy_x_position", 0) / max_x_pos,
                    info.get("enemy_y_position", 0) / max_y_pos
                ]
                obs_processed = tools.preprocess(obs)
                input_data = np.concatenate([obs_processed, pos_data])
                output = net.activate(input_data)

                action = [0] * len(env.buttons)
                for i, idx in enumerate(INDICES_BOTONES):
                    action[idx] = 1 if output[i] > 0.5 else 0

            for _ in range(exe_config.FRAME_SKIP):
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                frame_count += 1
                if done or frame_count >= max_frames:
                    break

                enemy_health = info.get("enemy_health", last_enemy_health)
                player_health = info.get("health", last_player_health)

                enemy_damage = max(0, last_enemy_health - enemy_health)
                self_damage = max(0, last_player_health - player_health)

                # --- Recompensas estilo ataque.py ---
                frame_window_counter += 1
                damage_acumulado_en_ventana += enemy_damage

                if frame_window_counter >= ventana_frames:
                    if damage_acumulado_en_ventana > 0:
                        fitness += damage_acumulado_en_ventana * ((damage_acumulado_en_ventana / ventana_frames) * 100)
                        if penalizaciones_seguidas > 0:
                            penalizaciones_seguidas -= 1
                    else:
                        fitness -= 50
                        penalizaciones_seguidas += 1

                    frame_window_counter = 0
                    damage_acumulado_en_ventana = 0

                if self_damage > 0:
                    fitness -= self_damage / 2

                if enemy_damage > 0:
                    fitness += enemy_damage * 1.8

                if penalizaciones_seguidas >= umbral_penalizaciones:
                    done = True

                last_enemy_health = enemy_health
                last_player_health = player_health

                if info.get("rounds_won", 0) != 0 or info.get("enemy_rounds_won", 0) != 0:
                    done = True
                    break

        # Suma el daño restante si la ventana no se completó
        if frame_window_counter > 0:
            if damage_acumulado_en_ventana > 0:
                fitness += damage_acumulado_en_ventana * ((damage_acumulado_en_ventana / ventana_frames) * 100)
            else:
                fitness -= 50

        env.close()
        avg_fitness[i_escn] = fitness

    genome.fitness = sum(avg_fitness) / len(avg_fitness)
    fitness_range = (-120, 120)
    tools.print_genoma_eval(genome, avg_fitness, fitness_range)
    return genome.fitness
