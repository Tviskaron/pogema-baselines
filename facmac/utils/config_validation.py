from typing import Union

from pogema import GridConfig, ExtraHard64x64, ExtraHard32x32, ExtraHard16x16, ExtraHard8x8, Hard8x8, Hard64x64, \
    Normal64x64, Easy64x64, Hard32x32, Normal32x32, Easy32x32, Hard16x16, Normal16x16, Easy16x16, Normal8x8, Easy8x8
from pydantic import BaseModel
from typing_extensions import Literal


class MADDPG(BaseModel):
    name: str = "maddpg_smac"

    action_range: float = None
    action_selector: str = "gumbel"
    epsilon_start: float = 0.5
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = 50000
    agent: str = "qmixrnn"
    obs_agent_id: bool = True  # Include the agent's one_hot id in the observation
    obs_last_action: bool = True  # Include the agent's last action (one_hot) in the observation
    agent_output_type: str = "pi_logits"
    batch_size_run: int = 1
    batch_size: int = 32
    buffer_size: int = 5000
    buffer_warmup: int = 0
    discretize_actions: bool = False
    double_q: bool = False
    epsilon_decay_mode: str = None
    exploration_mode: str = "gaussian"
    start_steps: int = 0  # Number of steps for uniform-random action selection, before running real policy. Helps exploration.
    act_noise: float = 0.1  # Stddev for Gaussian exploration noise added to policy at training time.
    ou_theta: float = 0.15  # D
    ou_sigma: float = 0.2  # D
    ou_noise_scale: float = 0.3
    final_ou_noise_scale: float = 0.
    gamma: float = 0.99
    grad_norm_clip: float = 10
    learner: str = "maddpg_learner_discrete"
    learn_interval: int = 1
    lr: float = 0.0025
    critic_lr: float = 0.0005
    td_lambda: float = 0.8
    critic_train_reps: int = 1
    q_nstep: int = 0  # 0 corresponds to default Q, 1 is r + gamma*Q, etc
    mac: str = "basic_mac"

    n_runners: str = None
    n_train: int = 1
    optimizer: str = "adam"  # D
    ou_stop_episode: int = 100  # training noise goes to zero after this episode
    rnn_hidden_dim: int = 64
    run_mode: str = None
    runner: str = "episode"
    runner_scope: str = 'episodic'
    target_update_interval: int = 200
    target_update_mode: str = 'hard'
    target_update_tau: float = 0.001
    test_greedy: bool = True
    test_interval: int = 10000
    test_nepisode: int = 32
    testing_on: bool = True
    verbose: bool = False
    weight_decay: bool = True
    weight_decay_factor: float = 0.0001

    agent_return_logits: bool = False
    q_embed_dim: int = 1
    mask_before_softmax: bool = True
    recurrent_critic: bool = False


class FACMAC(MADDPG):
    name: str = "facmac_smac"

    learner: str = "facmac_learner_discrete"
    mixer: str = "qmix"
    mixing_embed_dim: int = 32
    skip_connections: bool = False
    gated: bool = False
    hypernet_layers: int = 2
    hypernet_embed: int = 64
    hyper_initialization_nonzeros: int = 0


class PymarlSettings(BaseModel):
    env = "pogema"  # Environment name
    log_interval: int = 2000  # Log summary of stats after every {} timesteps
    runner_log_interval: int = 2000  # Log runner stats (not test stats) every {} timesteps
    learner_log_interval: int = 2000  # Log training stats every {} timesteps
    t_max: int = 10000000  # Stop running after this many timesteps
    use_cuda: bool = True  # Use gpu by default unless it isn't available
    buffer_cpu_only: bool = True  # If true we won't keep all of the replay buffer in vram


class Logging(BaseModel):
    use_tensorboard: bool = True  # Log results to tensorboard
    save_model: bool = True  # Save the models to disk
    save_model_interval: int = 10000000  # Save models after this many timesteps
    checkpoint_path: str = ""  # Load a checkpoint from this path
    evaluate: bool = False  # Evaluate model for test_nepisode episodes and quit (no training)
    load_step: int = 0  # Load model trained on this many timesteps (0 if choose max possible)
    save_replay: bool = False  # Saving the replay of the model loaded from checkpoint_path
    local_results_path: str = "results"  # Path for local results
    use_wandb: bool = True
    project_name: str = 'pogema-pymarl'


class Environment(BaseModel):
    grid_config: Union[
        Easy8x8, Normal8x8, Hard8x8, ExtraHard8x8,
        Easy16x16, Normal16x16, Hard16x16, ExtraHard16x16,
        Easy32x32, Normal32x32, Hard32x32, ExtraHard32x32,
        Easy64x64, Normal64x64, Hard64x64, ExtraHard64x64,
        GridConfig] = GridConfig()


class AlgoSettings(BaseModel):
    MADDPG: MADDPG = MADDPG()
    FACMAC: FACMAC = FACMAC()


class ExperimentConfig(BaseModel):
    env_args: Environment = Environment()
    logging: Logging = Logging()
    settings: PymarlSettings = PymarlSettings()
    algo_settings: AlgoSettings = AlgoSettings()
    algo: Literal["MADDPG", "FACMAC"] = "FACMAC"
