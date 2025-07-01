import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from torch.utils.tensorboard import SummaryWriter
from mk_env import MortalKombatEnv
from stable_baselines3.common.callbacks import CallbackList
from reward_logger_callback import RewardLoggerCallback
from datetime import datetime

def main():
    # Verifica el entorno personalizado
    #check_env(MortalKombatEnv(), warn=True)

    num_procesos = min(multiprocessing.cpu_count(), 1)
    print(f"Usando {num_procesos} procesos para entrenar...")

    # Crear carpetas si no existen
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = SubprocVecEnv([lambda: MortalKombatEnv() for _ in range(num_procesos)])

    # Hiperparámetros ajustables
    learning_rate = 2e-4          
    ent_coef = 0.015               # más entropía = más exploración (útil para agresividad)
    clip_range = 0.15             # más conservador para evitar updates inestables
    gamma = 0.99                  # descuento normal (puede bajarse un poco si quieres decisiones más inmediatas)
    gae_lambda = 0.95             # acorta la ventaja, puede ayudar si la señal es muy variable
    vf_coef = 0.4 

    # Ruta del modelo base
    model_path = "ppo_mk_model"
    log_name = datetime.now().strftime("PPO_%d-%m___%H-%M")

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
            gamma = gamma,             
            gae_lambda = gae_lambda, 
            vf_coef = vf_coef,
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
            gamma = gamma,
            gae_lambda = gae_lambda, 
            vf_coef = vf_coef,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )

    # Callback para checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_mk_checkpoint"
    )
    reward_logger_callback = RewardLoggerCallback()
    callback_list = CallbackList([checkpoint_callback, reward_logger_callback])

    print("🚀 Comenzando entrenamiento...")
    model.learn(
        total_timesteps=500_000*num_procesos,
        callback=callback_list,
        reset_num_timesteps=False,
        tb_log_name=log_name
    )

    # Guardar modelo final
    model.save(model_path)
    print(f"✅ Entrenamiento finalizado y modelo guardado como {model_path}.zip")

if __name__ == '__main__':
    main()
