import lifes_laws.config as exe_config
from . import tools
import neat
import retro
import numpy as np
import math
import random
from collections import deque

"""
De lo más importante del NEAT, la evaluacion de los genomas.
"""
def eval_genome(genome, config):

    avg_fitness = [120] * exe_config.CANTIDAD_MAPAS_A_ENTRENAR

    for i_escn in range(exe_config.CANTIDAD_MAPAS_A_ENTRENAR):
        # Aqui puedes ajustar el mapa segun encuentres necesario.
        # Los mapas faciles van del VeryEasy.LiuKang-02 al VeryEasy.LiuKang-15.
        # Los mapas dificiles tienen nombres personalizados.
        # Si solo vas a entrenar 1 mapa como con: escenario = f"LiuKangVsLiuKang_VeryHard_06", entonces es importante que CANTIDAD_MAPAS_A_ENTRENAR sea 1.
        # Si quieres entrenar un grupo de mapas al mismo tiempo por agente debes ajustar CANTIDAD_MAPAS_A_ENTRENAR al valor adecuado y hacer : 
        # escenario = f"VeryEasy.LiuKang-{(i_escn + 2):02d}" y 
        escenario = f"Level1.LiuKangVsJax.state"  # Este es el mapa de LiuKang rojo (nosotros) vs LiuKang azul (enemigo)
        env = retro.make(game='MortalKombatII-Genesis',state=escenario, render_mode=exe_config.RENDER_MODE)
        obs, info = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        action = [0] * len(exe_config.BOTONES_USADOS)

        # Inicializando variables.
        done = False
        frame_count = 0
        max_frames = 4500  # Un compate solamente
        warmup_frames = 200 # 15 frams despues de que pueden comenzar a pelear
        last_enemy_health = 120
        last_player_health = 120


        while not done and frame_count < max_frames:
            if frame_count < warmup_frames:         # 🔀 Acción aleatoria durante los warmup_frames, para que el agente siempre este en distintas situaciones
                pre_action = random.choice(list(exe_config.MOVIMIENTOS_BASE.keys()))
                action = list(pre_action)
                last_enemy_health = info.get("enemy_health", last_enemy_health)
                last_player_health = info.get("health", last_player_health)
            else:
                max_x_pos = 320.0   # Máxima posicion X en el mapa
                max_y_pos = 224.0   # Máxima posicion Y en el mapa
                
                # pos_data contiene la ubicación de del agente y del enemigo
                pos_data = [
                    info.get("x_position", 0) / max_x_pos,
                    info.get("y_position", 0) / max_y_pos,
                    info.get("enemy_x_position", 0) / max_x_pos,
                    info.get("enemy_y_position", 0) / max_y_pos
                ]
                obs_processed = tools.preprocess(obs)
                input_data = np.concatenate([obs_processed, pos_data]) # Juntamos la obs con pos_data
                output = net.activate(input_data)   # La activamos, para esto tuvimos que subir las neuronas input de 7056 a 7060

                for i in range(8):
                    action[i] = 1 if output[i] > 0.5 else 0

                

            # Mantenemos la acción por la cantidad de frames indicada en FRAME_SKIP. Esto hace hasta (FRAME_SKIP - 1) veces más rapido el entrenamiento
            for _ in range(exe_config.FRAME_SKIP):
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # También aumentamos el contador por cada frame saltado
                frame_count += 1
                if done or frame_count >= max_frames: 
                    break

                # 👇 Procesamiento de daño dentro del loop skip (opcional)
                enemy_health = info.get("enemy_health", last_enemy_health)
                player_health = info.get("health", last_player_health)

                enemy_damage = max(0, last_enemy_health - enemy_health)
                self_damage = max(0, last_player_health - player_health)
                

                avg_fitness[i_escn] += enemy_damage
                avg_fitness[i_escn] -= self_damage

                last_enemy_health = enemy_health
                last_player_health = player_health

                if info.get("rounds_won", 0) != 0 or info.get("enemy_rounds_won", 0) != 0:
                    done = True
                    break

        env.close()


    genome.fitness = (sum(avg_fitness) / len(avg_fitness))
    fitness_range = (0, 240) # Es ideal ajustar esto con el mínimo fitness posible y el máximo fitness posible, asi la representacion de los colores es fiel a la realidad.
    tools.print_genoma_eval(genome, avg_fitness, fitness_range)  

    return genome.fitness