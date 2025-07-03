import retro
import neat
import numpy as np
import cv2
import pickle
import random
import os
import math
import csv
import os
import neat
import glob
from datetime import datetime
from neat.parallel import ParallelEvaluator


BOTONES_USADOS = ['A', 'B', 'C', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']

#################################### Variables de entrenamiento ####################################

CANTIDAD_MAPAS_A_ENTRENAR = 5
GENERACIONES              = 100
CARPETA_CHECKPOINTS       = ""
NOMBRE_MEJOR_AGENTE       = ""
NUCLEOS                   = 0

####################################################################################################


def fitness_color(fitness, string=None) :
    RED = "\033[91m"
    ORANGE = "\033[38;5;208m"
    YELLOW = "\033[38;5;228m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"

    if fitness <= -80:
        color = RED
    elif fitness <= -40:
        color = ORANGE
    elif fitness <= 0:
        color = YELLOW
    elif fitness <= 40:
        color = GREEN
    elif fitness <= 80:
        color = CYAN
    else:
        color = BLUE
    if string != None:
        return f"{color}{string}\033[0m"
    return f"{color}{fitness:^8.2f}\033[0m"

def print_genoma_eval(genome, allfitness):
    print(f"\r🧬 Genoma {genome.key}, obtuvo: {fitness_color(genome.fitness)} " + " ".join(fitness_color(f, string="■") for f in allfitness))



class LastCheckpointSaver(neat.Checkpointer):
    def __init__(self, carpeta, frecuencia=1):
        self.carpeta = carpeta
        super().__init__(generation_interval=frecuencia, filename_prefix=os.path.join(carpeta, "neat-checkpoint-"))

    def save_checkpoint(self, config, population, species_set, generation):
        super().save_checkpoint(config, population, species_set, generation)


class ReporterCSV(neat.reporting.BaseReporter):
    def __init__(self, carpeta_csv):
        self.carpeta_csv = carpeta_csv
        self._inicializar_csvs()
        self.gen_actual = 0

    def _inicializar_csvs(self):
        os.makedirs(self.carpeta_csv, exist_ok=True)
        self.arch_genomas = os.path.join(self.carpeta_csv, "genomas.csv")
        self.arch_mejores = os.path.join(self.carpeta_csv, "mejores_genomas.csv")
        self.arch_especies = os.path.join(self.carpeta_csv, "especies.csv")

        if not os.path.exists(self.arch_genomas):
            with open(self.arch_genomas, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Generacion", "GenomaID", "Fitness", "Nodos", "Conexiones", "EspecieID"])

        if not os.path.exists(self.arch_mejores):
            with open(self.arch_mejores, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Generacion", "Fitness", "Nodos", "Conexiones", "EspecieID"])

        if not os.path.exists(self.arch_especies):
            with open(self.arch_especies, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Generacion", "EspecieID", "Tamaño", "Edad"])

    def end_generation(self, config, population, species_set):
        for gid, g in population.items():
            if g.fitness is not None:
                evaluados = [g for g in population.values() if g.fitness is not None]
                best = max(evaluados, key=lambda g: g.fitness) if evaluados else None


        with open(self.arch_genomas, "a", newline="") as f:
            writer = csv.writer(f)
            for gid, genome in population.items():
                if genome.fitness is not None:
                    especie_id = next((sid for sid, sp in species_set.species.items() if gid in sp.members), -1)
                    writer.writerow([self.gen_actual, gid, genome.fitness, len(genome.nodes), len(genome.connections), especie_id])

        with open(self.arch_mejores, "a", newline="") as f:
            especie_id = next((sid for sid, sp in species_set.species.items() if best.key in sp.members), -1)
            writer = csv.writer(f)
            writer.writerow([self.gen_actual, best.fitness, len(best.nodes), len(best.connections), especie_id])

        with open(self.arch_especies, "a", newline="") as f:
            writer = csv.writer(f)
            for sid, sp in species_set.species.items():
                writer.writerow([self.gen_actual, sid, len(sp.members), self.gen_actual - sp.created])

        self.gen_actual += 1


class HoraReporter(neat.reporting.BaseReporter):
        def start_generation(self, generation):
            print(f"🕑 Generacion {generation}, - Hora: {datetime.now().strftime('%H:%M')}")


# Esto sirve para procesar la imagen (los frames) y pueda ser interpretada
def preprocess(obs):
    image = obs[0]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (84, 84))
    image = image.astype(np.float32) / 255.0
    return image.flatten()

def distancia_funcion(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def fitness_funcion(agente_info, enemy_info):
    step_fitness = 0

    if enemy_info[0] != 0:
        step_fitness += enemy_info[0]*0.9
    
    if agente_info[0] != 0:
        step_fitness -= agente_info[0]*1.1

    return step_fitness

def eval_single_genome(genome, config):
    try:
        allfitness = [0]*CANTIDAD_MAPAS_A_ENTRENAR
        for i_escensarios in range(CANTIDAD_MAPAS_A_ENTRENAR):
            escenario = f"VeryEasy.LiuKang-{(i_escensarios + 2):02d}"

            env = retro.make(game='MortalKombatII-Genesis',state=escenario, render_mode=None)
            obs, _ = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            INDICES_BOTONES = [env.buttons.index(b) for b in BOTONES_USADOS]
            action = [0] * len(env.buttons)

            done = False
            frame_count = 0
            max_frames = 10000

            last_enemy_health = 120
            last_player_health = 120

            for _ in range(random.randint(5, 30)):
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    obs, _ = env.reset()

            while not done and frame_count < max_frames:
                obs_processed = preprocess(obs)
                output = net.activate(obs_processed)

                action = [0] * len(env.buttons)
                for i, idx in enumerate(INDICES_BOTONES):
                    action[idx] = 1 if output[i] > 0.5 else 0
                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                enemy_health = info.get("enemy_health", last_enemy_health)
                player_health = info.get("health", last_player_health)

                enemy_damage = max(0, last_enemy_health - enemy_health)
                self_damage = max(0, last_player_health - player_health)

                if info.get("rounds_won", 0) != 0:
                    done = True

                if info.get("enemy_rounds_won", 0) != 0:
                    done = True
                
                self_x = info.get("x_position", 0)
                self_y = info.get("y_position", 0)
                enemy_x = info.get("enemy_x_position", 0)
                enemy_y = info.get("enemy_y_position", 0)

                allfitness[i_escensarios] += fitness_funcion((self_damage, self_x, self_y), (enemy_damage, enemy_x, enemy_y))
                last_enemy_health = enemy_health
                last_player_health = player_health
                frame_count += 1

            env.close()

        genome.fitness = sum(allfitness) / len(allfitness)
        print_genoma_eval(genome, allfitness)
        return genome.fitness
    except Exception as e:
        print(f"Error evaluando genoma {genome.key}: {e}")
        genome.fitness = -10000  # asignar fitness bajo si hay error para que no quede None
        return genome.fitness


# Función principal de entrenamiento
def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    default = ""

    while default != "s" and default != "n":

        default = input("¿Quien cambiar la configuración default? (s/n): ").strip().lower()

        if default == "s":

            GENERACIONES = int(input("¿Cuantas generaciones quiere trabajar? (5-50): "))
            CARPETA = "checkpoints_v2"
            MEJOR_AGENTE = "mejor_agente_v2"

            # Asegurar carpeta checkpoints
            os.makedirs(CARPETA, exist_ok=True)

            NUCLEOS = int(input("¿Cuantos nucleos quiere trabajar? (1-10): "))

            # Preguntar si cargar desde checkpoint
            usar_checkpoint = input("¿Cargar checkpoint existente? (s/n): ").strip().lower() == "s"
            if usar_checkpoint:
                archivos = sorted([f for f in os.listdir(CARPETA) if f.startswith("neat-checkpoint-")])
                if archivos:
                    for i, f in enumerate(archivos):
                        print(f"{i+1}. {f}")
                    try:
                        idx = int(input("Selecciona el checkpoint a cargar: ")) - 1
                        checkpoint_path = os.path.join(CARPETA, archivos[idx])
                        print(f"Cargando {checkpoint_path}...")
                        population = neat.Checkpointer.restore_checkpoint(checkpoint_path)
                    except Exception as e:
                        print("Error al cargar checkpoint. Empezando desde cero.", e)
                        population = neat.Population(config)
                else:
                    print("No hay checkpoints disponibles. Empezando desde cero.")
                    population = neat.Population(config)
            else:
                print("Entrenamiento nuevo.")
                population = neat.Population(config)
        

    
    # Reportes
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix="checkpoints_v2"))

    pe = ParallelEvaluator(NUCLEOS, eval_single_genome)

    winner = population.run(pe.evaluate, n=GENERACIONES)

    print("\n--- Mejor genoma encontrado ---\n")

    with open(f"{MEJOR_AGENTE}.pkl", "wb") as f:
        pickle.dump(winner, f)

    print(f"Agente guardado en '{MEJOR_AGENTE}.pkl'")

if __name__ == "__main__":
    run("config-neat")

