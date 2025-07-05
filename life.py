import neat
import pickle
import os
import lifes_laws
from neat.parallel import ParallelEvaluator

"""
Incial el entrenamiento con neat
"""
def let_there_be_life(config_file = "config-neat"):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    
    # Pide las configuraciones del entrenamiento
    GENERACIONES, THREADS, NOMBRE_MEJOR_AGENTE, CARPETA_CHECKPOINTS, CARGAR_CHECKPOINT, CHECKPOINT  = lifes_laws.ask_config()

    # Configura el checkpoint inicial y donde se guardarán los siguientes.
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
    population.add_reporter(neat.Checkpointer(generation_interval=10, filename_prefix=f"{CARPETA_CHECKPOINTS}/generacion-")) # Se guardan checkpoints cada 10 generaciones.
    pe = ParallelEvaluator(THREADS, lifes_laws.eval_genome)

    # Iniciamos neat.
    winner = population.run(pe.evaluate, n=GENERACIONES)

    # Guardamos el mejor agente según lo que encontró neat.
    print("\n--- Mejor genoma hasta ahora encontrado ---\n")
    with open(f"{CARPETA_CHECKPOINTS}/{NOMBRE_MEJOR_AGENTE}.pkl", "wb") as f:
        pickle.dump(winner, f)
    print(f"Agente guardado en '{CARPETA_CHECKPOINTS}/{NOMBRE_MEJOR_AGENTE}.pkl'")


"""
Explica como usar todo.
"""
def help():
    lifes_laws.help()