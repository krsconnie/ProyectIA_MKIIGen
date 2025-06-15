# Instalaciones requeridas
# pip3 install opencv-python
# pip3 install neat

import retro
import neat
import numpy as np
import cv2
import pickle

def preprocess(obs):
    image = obs[0]
    
    # Si la imagen tiene 3 canales (RGB), convertir a escala de grises
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

        fitness = 0
        done = False
        frame_count = 0
        max_frames = 1000

        while not done and frame_count < max_frames:
            obs_processed = preprocess(obs)
            output = net.activate(obs_processed)
            action = [1 if o > 0.5 else 0 for o in output]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            fitness += reward
            frame_count += 1

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

    winner = population.run(eval_genomes, n=5)  # puedes aumentar n luego

    print("\n--- Mejor genoma encontrado ---\n")
    print(winner)

    with open("mejor_agente.pkl", "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    run("config-neat")

