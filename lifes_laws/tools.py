import cv2
import sys
import os
import numpy as np
import lifes_laws.config as config
import threading
import neat

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

Ejemplo con 5 mapas:
🧬 Genoma 230, fitness:   -76.09   ■ ■ ■ ■ ■
🧬 Genoma 232, fitness:  -114.34   ■ ■ ■ ■ ■
🧬 Genoma 237, fitness:   101.72   ■ ■ ■ ■ ■

Ejemplo con 2 mapas:
🧬 Genoma 230, fitness:   -76.09   ■ ■
🧬 Genoma 232, fitness:  -114.34   ■ ■
🧬 Genoma 237, fitness:   101.72   ■ ■
"""
def print_genoma_eval(genome, allfitness, min_max):
    def fitness_color(fitness, min_max, string=None) :
        RED = "\033[91m"
        ORANGE = "\033[38;5;208m"
        YELLOW = "\033[38;5;220m"
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        BLUE = "\033[94m"
        min, max = min_max
        seg = (max - min)/6

        if fitness <= (min + seg):
            color = RED
        elif fitness <= (min + 2*seg):
            color = ORANGE
        elif fitness <= (min + 3*seg):
            color = YELLOW
        elif fitness <= (min + 4*seg):
            color = GREEN
        elif fitness <= (min + 5*seg):
            color = CYAN
        else:
            color = BLUE
        if string != None:
            return f"{color}{string}\033[0m"
        return f"{color}{fitness:>9.2f}\033[0m"
    
    print(f"🧬 Genoma {genome.key}, fitness: {fitness_color(genome.fitness, min_max)} " + " ".join(fitness_color(f, min_max, string="■") for f in allfitness))


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
            if CARGAR_CHECKPOINT:
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
                    CARGAR_CHECKPOINT = False
                    print("No hay checkpoints disponibles. Se empezará desde cero.")
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
                    THREADS = input("¿Cuántos threads quieres utlizar? (1-16, 'd' para default): ")
                    if THREADS == 'd':
                        THREADS = config.THREADS
                        continue
                    THREADS = int(THREADS)
                    if THREADS < 1:
                        print("⚠️ Ingrese un valor valido.")
                except ValueError:
                    print("El valor debe ser un numero natural.")

            CANTIDAD_MAPAS_A_ENTRENAR = -1
            while CANTIDAD_MAPAS_A_ENTRENAR < 1:
                try:
                    CANTIDAD_MAPAS_A_ENTRENAR = input("¿Cuántos mapas quieres entrenar? (1-14, 'd' para default): ")
                    if CANTIDAD_MAPAS_A_ENTRENAR == 'd':
                        CANTIDAD_MAPAS_A_ENTRENAR = config.CANTIDAD_MAPAS_A_ENTRENAR
                        continue
                    CANTIDAD_MAPAS_A_ENTRENAR = int(CANTIDAD_MAPAS_A_ENTRENAR)
                    if CANTIDAD_MAPAS_A_ENTRENAR < 1:
                        print("⚠️ Ingrese un valor valido.")
                except ValueError:
                    print("El valor debe ser un numero natural.")

            FRAME_SKIP = -1
            while FRAME_SKIP < 1:
                try:
                    FRAME_SKIP = input("¿Cuánto de frame skip quieres que haya? (1-4, 'd' para default): ")
                    if FRAME_SKIP == 'd':
                        FRAME_SKIP = config.FRAME_SKIP
                        continue
                    FRAME_SKIP = int(FRAME_SKIP)
                    if FRAME_SKIP < 1:
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
    
"""
Explica el programa.
"""
def help():
    print("""
    🧬 \033[1mBienvenido a la evolución.\033[0m
          
    Te explico todo muy rápido.
          
    En \033[3mmain.py\033[0m encontrarás comentados y seccionados los principales métodos.
          
    El más importante es \033[32mlifes\033[0m.\033[33mlet_there_be_life()\033[0m, este inicia el entrenamiento según
    las configuraciones asignadas en config-neat y en \033[3mlifes_laws/config.py\033[0m.
    En este último están todas las configuraciones realacionadas a los entrenamientos: cuantos hilos a la vez, 
    cuantas generaciones, el frame skip, donde se gaurdaran los checkpoints, entre otras.
    
    Puedes configurarlas ahi, aunque cuando ejecutes \033[32mlifes\033[0m.\033[33mlet_there_be_life()\033[0m, este te 
    preguntará si quieres usar las configuraciones default o no. Si marcas que no, te hara preguntas sobre las configuraciones 
    para asignarlas.
          
    Por otro lado tenemos a \033[32mlifes\033[0m.\033[33mjugar_humano()\033[0m con ella poder jugar tu encontra de los bots base 
    del juego. Esto es util para encontrar problemas, identificar puntos criticos, etc.
        
    Tambien esta \033[32mlifes\033[0m.\033[33mjugar_agente()\033[0m, como es de esperar, con esta podras utilizar el mejor agente 
    guardado hasta la fecha, deberas espesificarlo.

    Si lo que quieres es ajustar el fitness, deberas ir a \033[3mlifes_laws/fitness.py\033[0m, ahí encontraras la que probablemente es 
    la función más importante de todo. Si la lees un poco, encontrarás la seccion donde debes ajustar el fitness.
          
    Creo que eso es todo, muchas gracias por leer y espero que les sea de utilidad. ✨
    
          
""")