# 🧬 Bienvenido a la evolución

Este proyecto permite entrenar agentes usando NEAT en el juego Mortal Kombat II (versión Genesis),  
y también jugar como humano para pruebas o comparaciones.

### 📦 Archivos principales
   
- `life.py`: Entrena al agente.  
- `lifes_laws/fitness.py`: Contiene la funcion que evalua a los genomas.  
- `test.py`: Contiene los test para probar el agente o otras funciónes.
- `config-neat`: Configuración para el algoritmo NEAT.  
- `lifes_laws/config.py`: Configuración del entrenamiento. 

### 🚀 Cómo ejecutar

   Antes que nada, recomiendo ejecutar en `main.py` la funcion `life.help()` y asi entender un poco más todo.
    Para hacerlo nomas debe ejecutar en terminal el sigueitne comando, la función `life.help()` ya esta ahí.
   ```bash
   python -m main
   ```


### 🎯 Princiaples requisitos

- Python 3.10+  
- `pygame`  
- `stable-retro`  
- `neat-python`

> 💾 **Nota:** El juego **'MortalKombatII-Genesis'** debe estar instalado y configurado en `gym-retro`.
