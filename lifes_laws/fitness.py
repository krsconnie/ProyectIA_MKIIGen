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

    avg_fitness = [0] * exe_config.CANTIDAD_MAPAS_A_ENTRENAR

    for i_escn in range(exe_config.CANTIDAD_MAPAS_A_ENTRENAR):
        escenario = f"LiuKangVsLiuKang_VeryHard_06"

        env = retro.make(game='MortalKombatII-Genesis',state=escenario, render_mode=exe_config.RENDER_MODE)
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

        while not done and frame_count < max_frames:
            if frame_count < warmup_frames:
                # 🔀 Acción aleatoria durante los primeros frames
                action = env.action_space.sample()
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

                # 👇 También aumentamos el contador por cada frame saltado
                frame_count += 1
                if done or frame_count >= max_frames:
                    break

                # 👇 Procesamiento de daño dentro del loop skip (opcional)
                enemy_health = info.get("enemy_health", last_enemy_health)
                player_health = info.get("health", last_player_health)

                enemy_damage = max(0, last_enemy_health - enemy_health)
                self_damage = max(0, last_player_health - player_health)

                if enemy_damage != 0:
                    avg_fitness[i_escn] += enemy_damage

                # if self_damage != 0:
                #     avg_fitness[i_escn] -= self_damage * 1.1

                last_enemy_health = enemy_health
                last_player_health = player_health

                if info.get("rounds_won", 0) != 0 or info.get("enemy_rounds_won", 0) != 0:
                    done = True
                    break

        env.close()

    genome.fitness = sum(avg_fitness) / len(avg_fitness)
    tools.print_genoma_eval(genome, avg_fitness)
    return genome.fitness