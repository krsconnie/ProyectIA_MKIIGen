import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from mk_env import MortalKombatEnv

def main():
    num_procesos = min(multiprocessing.cpu_count(), 2)
    print(f"Usando {num_procesos} procesos para entrenar...")

    # Crear carpetas si no existen
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = SubprocVecEnv([lambda: MortalKombatEnv() for _ in range(num_procesos)])

    # Hiperparámetros ajustables
    learning_rate = 2e-4
    ent_coef = 0.01
    clip_range = 0.2

    # Ruta del modelo base
    model_path = "ppo_mk_model"

    if os.path.exists(f"{model_path}.zip"):
        print("📦 Cargando modelo base existente con nuevos hiperparámetros...")

        # Paso 1: cargar el modelo viejo
        old_model = PPO.load(model_path)

        # Paso 2: crear nuevo modelo con nuevos parámetros
        new_model = PPO(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            clip_range=clip_range,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )

        # Paso 3: transferir pesos
        new_model.policy.load_state_dict(old_model.policy.state_dict())
        model = new_model
    else:
        print("✨ Creando modelo nuevo...")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            clip_range=clip_range,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )

    # Callback para checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_mk_checkpoint"
    )

    print("🚀 Comenzando entrenamiento...")
    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        tb_log_name="run",
        reset_num_timesteps=False
    )

    # Guardar modelo final
    model.save(model_path)
    print(f"✅ Entrenamiento finalizado y modelo guardado como {model_path}.zip")

if __name__ == '__main__':
    main()
