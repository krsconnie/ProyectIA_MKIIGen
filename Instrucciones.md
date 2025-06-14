# Instrucciones #

Cabe aclarar que estas instrucciones toman en consideración que se trabaja en linux.

## Requisitos previos:

- Python 3.6 a 3.10  
- `pip3` instalado  
- Sistema operativo de **64 bits**  
- Git

**Stable Retro no es compatible con sistemas de 32 bits.**

Se recomienda usar un entorno virtual (opcional):

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Sobre el ROM:

El ROM de **Mortal Kombat II (Genesis)** no está incluido por defecto. Debes conseguirlo por tu cuenta desde una fuente segura.

Se recomienda buscarlo en el megathread de [r/Roms](https://www.reddit.com/r/Roms), donde se enlazan colecciones confiables y actualizadas.

### Requisitos del ROM:

- Debe ser de la colección **No-Intro**.
- Debe tener la extensión `.md` o `.gen` (Sega Genesis / Mega Drive).
- Puede estar comprimido en `.zip`, lo cual es aceptado por Stable Retro.
- Nombre típico del archivo:

```
Mortal Kombat II (USA).zip
```

Stable Retro usa los **hashes SHA-1** de No-Intro para identificar ROMs válidos. Si el ROM no es de No-Intro (por ejemplo, si tiene una intro pirateada o una modificación), **no será reconocido** al momento de importarlo.


## Proceso:

1. Clona el repositorio de Stable Retro:

```bash
git clone https://github.com/openai/stable-retro.git
cd stable-retro
```

2. Instala Stable Retro:

```bash
pip3 install stable-retro
```

3. Importar el ROM a stable Retro

Desde la terminal asegurate de estar ubicado en stable-retro/retro/data/stable/MortalKombatII-Genesis y ejecuta:

```bash
python3 -m retro.import ./ Ubicación del archivo(path) /
```

Esto procesará el archivo y lo copiará al directorio interno de integración, siempre que el hash del ROM coincida con el esperado.

**Si esto no funciona**, intenta descomprimir el .zip e intentarlo denuevo con el archivo .md, si esto tampoco funciona renombra el archivo y cambiale la extensión a .gen 

---

4. Verificar si el juego se importó:

De haber realizado correctamente el paso anterior la terminal debería arrojar el mensaje:

```
Importing MortalKombatII-Genesis
Imported 1 games
```

5.Procesos relacionados con nuestro proyecto unu
