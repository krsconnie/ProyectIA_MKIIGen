# Instalaciones requeridas
# pip3 install opencv-python
# pip3 install neat

# La variable info tiene la siguiente estructura: enemy_health, health, enemy_rounds_won, rounds_won 
# Esos atributos se podrían tener en cuenta para entrenarlo de una u otra forma
# La lista de botones es la env.buttons es ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
# Al editar la lista de botones disponibles, en config-neat se debe modificar la variable num_outputs
# Si se quiere cambiar la cantidad de genomas/agentes por generaciones, en config-neat se cambia el atributo pop_size
# Si se quiere cambiar la cantidad de generaciones, en la función run hay una variable para cambiar eso

import retro
import neat
import numpy as np
import cv2
import pickle
import random
from neat import ParallelEvaluator

# Botones a utilizar para el agente, se omiten botones MODE, X, Y, Z ya que no se usan para pelear
# Los botones que se usan son los de abajo, el control de genesis PUEDE tener más botones, pero el mortal kombat,
# está funcionando con el control básico, en este pdf salen los controles https://classicreload.com/sites/default/files/genesis-mortal-kombat-ii-Manual.pdf 
BOTONES_USADOS = ['A', 'B', 'C', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']
#BOTONES_USADOS = ['B']

# Esto sirve para procesar la imagen (los frames) y pueda ser interpretada
def preprocess(obs):
    image = obs[0]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (84, 84))
    image = image.astype(np.float32) / 255.0
    return image.flatten()


# Evaluación de UN solo genoma (para que pueda ser paralelizada)
def evaluate_single_genome(genome, config):
    with open('archivo.txt', 'a') as archivo:
        archivo.write(f"Evaluando Genoma ID: {genome.key}\n")

    env = retro.make(game='MortalKombatII-Genesis', state="Level1.LiuKangVsJax")
    obs = env.reset()[0]
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Mapeo de índices de botones usados
    INDICES_BOTONES = [env.buttons.index(b) for b in BOTONES_USADOS]

    done = False
    frame_count = 0
    max_frames = 10000

    # Variables para recompensas
    fitness = 0
    last_enemy_health = 120
    last_player_health = 120

    # Aleatorizar un poco el inicio para romper simetrías
    for _ in range(random.randint(5, 30)):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs = env.reset()[0]

    # Variables adicionales para recompensas más complejas
    ventana_frames = 180
    frame_window_counter = 0
    damage_acumulado_en_ventana = 0
    penalizaciones_seguidas = 0
    umbral_penalizaciones = 7

    # Mientras no esté listo y la cantidad de frames máximo no se haya superado
    while not done and frame_count < max_frames:
        obs_processed = preprocess(obs)
        output = net.activate(obs_processed)

        # Resetear vector de acciones
        action = [0] * len(env.buttons)
        for i, idx in enumerate(INDICES_BOTONES):
            action[idx] = 1 if output[i] > 0.5 else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        enemy_damage = last_enemy_health - info['enemy_health']
        self_damage = last_player_health - info['health']

        # Condiciones adicionales de término
        if info['enemy_health'] <= 0 or info['health'] <= 0:
            done = True

        if info['rounds_won'] > 0 or info['enemy_rounds_won'] > 0:
            done = True

        frame_window_counter += 1
        damage_acumulado_en_ventana += enemy_damage

        if frame_window_counter >= ventana_frames:
            if damage_acumulado_en_ventana > 0:
                fitness += damage_acumulado_en_ventana * ((damage_acumulado_en_ventana/ventana_frames)*100)
                if penalizaciones_seguidas > 0:
                    penalizaciones_seguidas -= 1
            else:
                fitness -= 50
                penalizaciones_seguidas += 1

            frame_window_counter = 0
            damage_acumulado_en_ventana = 0

        if self_damage > 0:
            fitness -= self_damage

        if enemy_damage > 0:
            fitness += enemy_damage

        if penalizaciones_seguidas >= umbral_penalizaciones:
            done = True

        last_enemy_health = info['enemy_health']
        last_player_health = info['health']

        frame_count += 1

    # Evaluar el daño restante al final del round
    if frame_window_counter > 0:
        if damage_acumulado_en_ventana > 0:
            fitness += damage_acumulado_en_ventana
        else:
            fitness -= 50

    print({fitness})
    genome.fitness = fitness
    env.close()


# Función principal de entrenamiento
def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Aquí usamos ParallelEvaluator con 4 procesos (ajusta según tu CPU)
    num_workers = 5
    pe = ParallelEvaluator(num_workers, evaluate_single_genome)
    
    for genome_id, genome in population.population.items():
        print(f"[CHECK] Genoma {genome_id} fitness inicial: {genome.fitness}")

    winner = population.run(pe.evaluate, n=50)  # n define cuántas generaciones se harán
    
    for genome_id, genome in population.population.items():
        print(f"Genoma {genome_id} -> fitness: {genome.fitness}")


    print("\n--- Mejor genoma encontrado ---\n")
    print(winner)

    with open("mejor_agente.pkl", "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    run("config-neat")
