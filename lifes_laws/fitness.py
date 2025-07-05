import lifes_laws.config as exe_config
from . import tools
import neat
import retro
import numpy as np
import random

"""
Asigna el fitness
"""
def fitness_funtion(tiempo_vivo, combinaciones, total_botones, spam_penalty, repeticiones):

    promedio_botones = total_botones / (tiempo_vivo/exe_config.FRAME_SKIP)

    fitness = (
        len(combinaciones)
        + tiempo_vivo / (600/exe_config.FRAME_SKIP)
        - spam_penalty
        - repeticiones * 0.2
        - promedio_botones * 0.05
    )

    return max(fitness, 0)  # evitamos fitness negativos

"""
De lo más importante del NEAT, la evaluacion de los genomas.
"""
def eval_genome(genome, config):

    avg_fitness = [0] * exe_config.CANTIDAD_MAPAS_A_ENTRENAR

    for i_escn in range(exe_config.CANTIDAD_MAPAS_A_ENTRENAR):
        # Aqui puedes ajustar el mapa segun encuentres necesario.
        # Los mapas faciles van del VeryEasy.LiuKang-02 al VeryEasy.LiuKang-15.
        # Los mapas dificiles tienen nombres personalizados.
        # Si solo vas a entrenar 1 mapa como con: escenario = f"LiuKangVsLiuKang_VeryHard_06", entonces es importante que CANTIDAD_MAPAS_A_ENTRENAR sea 1.
        # Si quieres entrenar un grupo de mapas al mismo tiempo por agente debes ajustar CANTIDAD_MAPAS_A_ENTRENAR al valor adecuado y hacer : 
        # escenario = f"VeryEasy.LiuKang-{(i_escn + 2):02d}" y 
        escenario = f"VeryEasy.LiuKang-07"  # Este es el mapa de LiuKang rojo (nosotros) vs LiuKang azul (enemigo)
        env = retro.make(game='MortalKombatII-Genesis',state=escenario, render_mode=exe_config.RENDER_MODE)
        obs, info = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        INDICES_BOTONES = [env.buttons.index(b) for b in exe_config.BOTONES_USADOS]
        action = [0] * len(env.buttons)

        # Inicializando variables.
        done = False
        frame_count = 0
        max_frames = 4500  # Un compate solamente
        warmup_frames = 200 # 15 frams despues de que pueden comenzar a pelear
        last_enemy_health = 120
        last_player_health = 120
        acciones = []
        combinaciones = set()
        last_action = [0] * len(env.buttons)
        total_botones = 0
        spam_penalty = 0
        repeticiones = 0
        promedio_botones = 0

        while not done and frame_count < max_frames:
            if frame_count < warmup_frames:         # 🔀 Acción aleatoria durante los warmup_frames, para que el agente siempre este en distintas situaciones
                action = env.action_space.sample()
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

                # Elegimos una acción
                action = [0] * len(env.buttons)
                for i, idx in enumerate(INDICES_BOTONES):
                    action[idx] = 1 if output[i] > 0.5 else 0
                
                suma = sum(action)
                combinaciones.add(tuple(action))
                acciones.append(action)
                total_botones += suma
                if suma > 3:
                    spam_penalty += (suma - 3) * 0.1
                if i > 0 and action == last_action:
                    repeticiones += 1
                last_action = action

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
                


                last_enemy_health = enemy_health
                last_player_health = player_health

                if info.get("rounds_won", 0) != 0 or info.get("enemy_rounds_won", 0) != 0:
                    done = True
                    break

        

        avg_fitness[i_escn] = fitness_funtion(frame_count - warmup_frames, combinaciones=combinaciones, total_botones=total_botones, repeticiones=repeticiones)

        env.close()

    genome.fitness = sum(avg_fitness) / len(avg_fitness)
    fitness_range = (0, 1000) # Es ideal ajustar esto con el mínimo fitness posible y el máximo fitness posible, asi la representacion de los colores es fiel a la realidad.
    tools.print_genoma_eval(genome, avg_fitness, fitness_range)  
    return genome.fitness