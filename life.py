import neat
import pickle
import os
import lifes_laws
from neat.parallel import ParallelEvaluator




def let_there_be_life(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    
    GENERACIONES, THREADS, NOMBRE_MEJOR_AGENTE, CARPETA_CHECKPOINTS, CARGAR_CHECKPOINT, CHECKPOINT  = lifes_laws.ask_config()

    # Asegurar carpeta checkpoints
    os.makedirs(CARPETA_CHECKPOINTS, exist_ok=True)
    if CARGAR_CHECKPOINT:
        checkpoint_path = os.path.join(CARPETA_CHECKPOINTS, CHECKPOINT)
        print(f"Cargando {checkpoint_path}...")
        population = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        population = neat.Population(config)

    
    # Reportes
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=f"{CARPETA_CHECKPOINTS}/generacion-"))

    pe = ParallelEvaluator(THREADS, lifes_laws.eval_genome)

    winner = population.run(pe.evaluate, n=GENERACIONES)

    print("\n--- Mejor genoma encontrado ---\n")

    with open(f"{CARPETA_CHECKPOINTS}/{NOMBRE_MEJOR_AGENTE}.pkl", "wb") as f:
        pickle.dump(winner, f)

    print(f"Agente guardado en '{NOMBRE_MEJOR_AGENTE}.pkl'")

if __name__ == "__main__":
    let_there_be_life("config-neat")

