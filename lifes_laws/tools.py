import cv2
import sys
import os
import numpy as np
from . import config

"""
Reajusta el tamaño de la imagen a 84 x 84, para que sea más liviana de procesar.
"""
def preprocess(obs):
    image = obs[0]
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (84, 84))
    image = image.astype(np.float32) / 255.0
    return image.flatten()


"""
Imprime en terminal el resultado de los genomas mientras se evaluan. 

Los colores van desde el rojo (muy bajo), hasta el azul (muy alto).
En un numero se representa el fitness promedio de los mapas y en cuadrados 
los resultados individuales de cada mapa.

Ejemplo:
🧬 Genoma 230, fitness:   -76.09   ■ ■ ■ ■ ■
🧬 Genoma 232, fitness:  -114.34   ■ ■ ■ ■ ■
🧬 Genoma 237, fitness:   101.72   ■ ■ ■ ■ ■
"""
def print_genoma_eval(genome, allfitness):
    def fitness_color(fitness, string=None) :
        RED = "\033[91m"
        ORANGE = "\033[38;5;208m"
        YELLOW = "\033[38;5;220m"
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
        return f"{color}{fitness:>9.2f}\033[0m"
    
    print(f"🧬 Genoma {genome.key}, fitness: {fitness_color(genome.fitness)} " + " ".join(fitness_color(f, string="■") for f in allfitness))


"""
Pregunta por las configuraciones de ejecución.
"""
def ask_config():
    default_options = ""

    GENERACIONES        = None
    THREADS             = None
    NOMBRE_MEJOR_AGENTE = None
    CARPETA_CHECKPOINTS = None
    CARGAR_CHECKPOINT   = None
    CHECKPOINT          = None

    while default_options != "y" and default_options != "n":

        default = input("¿Quieres usar la configuración default indicada en \"\\lifes_laws\\config.py\"? (y/n): ").strip().lower()

        if default == "y":
            GENERACIONES        = config.GENERACIONES
            THREADS             = config.THREADS
            NOMBRE_MEJOR_AGENTE = config.NOMBRE_MEJOR_AGENTE
            CARPETA_CHECKPOINTS = config.CARPETA_CHECKPOINTS
            CARGAR_CHECKPOINT   = config.CARGAR_CHECKPOINT

        elif default == "n":
            GENERACIONES = -1
            while GENERACIONES < 1:
                try:
                    GENERACIONES = input("¿Cuántas generaciones quieres realizar? (>0, 'd' para default): ")
                    if GENERACIONES == 'd':
                        GENERACIONES = config.GENERACIONES
                        continue
                    GENERACIONES = int(GENERACIONES)
                    if GENERACIONES < 1:
                        print("⚠️ Ingrese un valor valido.")
                except ValueError:
                    print("El valor debe ser un numero natural.")
            
            THREADS = -1
            while THREADS < 1:
                try:
                    THREADS = input("¿Cuántos threads quieres utlizar? (>0, 'd' para default): ")
                    if THREADS == 'd':
                        THREADS = config.THREADS
                        continue
                    THREADS = int(THREADS)
                    if THREADS < 1:
                        print("⚠️ Ingrese un valor valido.")
                except ValueError:
                    print("El valor debe ser un numero natural.")
            
            
            NOMBRE_MEJOR_AGENTE = ""
            while NOMBRE_MEJOR_AGENTE == "":
                NOMBRE_MEJOR_AGENTE = input("¿Nombre del archivo que tendrá al mejor agente? ('d' para default): ")
                if NOMBRE_MEJOR_AGENTE == 'd':
                    NOMBRE_MEJOR_AGENTE = config.NOMBRE_MEJOR_AGENTE
                    continue
                if NOMBRE_MEJOR_AGENTE == "":
                    print("⚠️ Ingrese un nombre valido.")

            CARPETA_CHECKPOINTS = ""
            while CARPETA_CHECKPOINTS == "":
                CARPETA_CHECKPOINTS = input("¿Nombre de la carpeta que almacenará los checkpoints? ('d' para default): ")
                if CARPETA_CHECKPOINTS == 'd':
                    CARPETA_CHECKPOINTS = config.CARPETA_CHECKPOINTS
                    continue
                if CARPETA_CHECKPOINTS == "":
                    print("⚠️ Ingrese un nombre valido.")

            
            CARGAR_CHECKPOINT = False
            CHECKPOINT = None
            confirm_CARGAR_CHECKPOINT = ""
            while confirm_CARGAR_CHECKPOINT != "y" and confirm_CARGAR_CHECKPOINT != "n":
                confirm_CARGAR_CHECKPOINT = input("¿Desea cargar desde alguna generación existente? (y/n): ").strip().lower()
                if confirm_CARGAR_CHECKPOINT == "y":
                    os.makedirs(CARPETA_CHECKPOINTS, exist_ok=True)
                    archivos = sorted([f for f in os.listdir(CARPETA_CHECKPOINTS) if f.startswith("generacion-")])
                    if archivos:
                        for i, f in enumerate(archivos):
                            print(f"{i}. {f}")
                        try:
                            idx = int(input("Selecciona la generación a cargar: "))
                            CARGAR_CHECKPOINT = True
                            CHECKPOINT = archivos[idx]
                        except Exception as e:
                            print("Error al cargar checkpoint.", e)
                            sys.exit(1)
                    else:
                        print("No hay checkpoints disponibles. Se empezará desde cero.")
                elif confirm_CARGAR_CHECKPOINT == "n":
                    print("✨ Nuevo entrenamiento.")
                else:
                    print("⚠️ Ingrese una opción valida.")
                    continue
        else:
            print("⚠️ Ingrese una opción valida.")
            continue

        print("🔝 Comenzando... Esto puede tardar unos segundos...")
        return (GENERACIONES, THREADS, NOMBRE_MEJOR_AGENTE, CARPETA_CHECKPOINTS, CARGAR_CHECKPOINT, CHECKPOINT)