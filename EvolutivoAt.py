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

# Botones a utilizar para el agente, se omiten botones MODE y START ya que no se usan para pelear
# Los botones que se usan son los de abajo, el control de genesis PUEDE tener más botones, pero el mortal kombat,
# está funcionando con el control básico, en este pdf salen los controles https://classicreload.com/sites/default/files/genesis-mortal-kombat-ii-Manual.pdf 
BOTONES_USADOS = ['A', 'B', 'C', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']
#BOTONES_USADOS = ['Z','X']

# Esto sirve para procesar la imagen (los frames) y pueda ser interpretada
def preprocess(obs):
    image = obs[0]
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (84, 84))
    image = image.astype(np.float32) / 255.0
    return image.flatten()


# Evaluación del fitness de cada genoma
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        env = retro.make(game='MortalKombatII-Genesis')
        obs = env.reset()[0]  # gymnasium retorna (obs, info)
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Mapeo de índices de botones usados
        INDICES_BOTONES = [env.buttons.index(b) for b in BOTONES_USADOS]
        action = [0] * len(env.buttons)

        done = False
        frame_count = 0
        max_frames = 10000 #determina cuantos frames va a durar la partida.

        # Variables que se irán teniendo en cuenta para dar recompensas
        fitness = 0 #número que determina cuan bueno es el agente. Esto es algo propio de NEAT
        last_enemy_health = 120
        last_player_health = 120


        # Aleatorizar un poco el inicio para romper simetrías. En entornos muy repetitivos (deterministas), el aprendizaje evolutivo puede teneder a repetir soluciones.
        # Con este ruido inicial, le da al agente un nuevo "escenario" en cada inicio de entrenamiento. 
        for _ in range(random.randint(5, 30)):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs = env.reset()[0]

        # Mientras no esté listo y la cantidad de frames máximo no se haya superado, no estoy muy seguro como funciona lo de done, por eso
        # Agregue lo del or para ser específico cuando debe acabar
        while not done and frame_count < max_frames or (info['enemy_rounds_won'] == 3) or (info['rounds_won'] == 3):
            obs_processed = preprocess(obs)
            output = net.activate(obs_processed)

            # Resetear vector de acciones, el vector action tendrá valores entre 0 y 1, si es mayor a 0.5 se interpreta como que se presiona el botón.
            #Esto implica que el agente puede presionar varios botones a la vez.
            action = [0] * len(env.buttons)
            for i, idx in enumerate(INDICES_BOTONES):
                action[idx] = 1 if output[i] > 0.5 else 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            enemy_damage = last_enemy_health - info['enemy_health']
            self_damage = last_player_health - info['health']

            # Fitness: premia el daño hecho y penaliza daño recibido, recompesas básicas por ahora
            fitness += enemy_damage * 2 - self_damage

            # Bonus por ganar rounds
            fitness += info['rounds_won'] * 50

            frame_count += 1


            # print(info)

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

    winner = population.run(eval_genomes, n=5)  # n define cuantas generaciones se harán

    print("\n--- Mejor genoma encontrado ---\n")
    print(winner)

    with open("mejor_agente.pkl", "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    run("config-neat")

