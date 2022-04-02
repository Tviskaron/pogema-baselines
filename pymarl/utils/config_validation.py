from pogema import GridConfig
from pydantic import BaseModel
from typing_extensions import Literal


class DQNBase(BaseModel):
    action_selector: str = "epsilon_greedy"
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = 50000
    runner: str = "episode"
    buffer_size: int = 5000
    target_update_interval: int = 200
    agent_output_type: str = "q"
    learner: str = "q_learner"
    double_q: bool = True

    name: str = None


class QMIX(DQNBase):
    name: str = "qmix"
    mixer: str = "qmix"
    mixing_embed_dim: int = 32
    hypernet_layers: int = 2
    hypernet_embed: int = 64


class VDN(DQNBase):
    name: str = "vdn"
    mixer: str = "vdn"


class QTRAN(DQNBase):
    name: str = "qtran"

    learner: str = "qtran_learner"
    mixer: str = "qtran_base"
    mixing_embed_dim: int = 64
    qtran_arch: str = "qtran_paper"

    opt_loss: int = 1
    nopt_min_loss: float = 0.1
    network_size: str = "small"


class IQL(DQNBase):
    name: str = "iql"
    mixer: str = None


class COMA(DQNBase):
    name: str = "coma"
    action_selector: str = "multinomial"

    epsilon_start: float = 0.5
    epsilon_finish: float = 0.01
    epsilon_anneal_time: int = 100000
    mask_before_softmax: bool = False

    runner: str = "parallel"
    buffer_size: int = 1
    batch_size_run: int = 1
    batch_size: int = 1

    mixer: str = None


class PymarlSettings(BaseModel):
    mac = "basic_mac"  # Basic controller
    env = "pogema"  # Environment name
    batch_size_run: int = 1  # Number of environments to run in parallel
    test_nepisode: int = 0  # Number of episodes to test for
    test_interval: int = 2000  # Test after {} timesteps have passed
    test_greedy: bool = True  # Use greedy evaluation (if False, will set epsilon floor to 0
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


class RlParams(BaseModel):
    gamma: float = 0.99
    batch_size: int = 32  # Number of episodes to train on
    lr: float = 0.0005  # Learning rate for agents
    critic_lr: float = 0.0005  # Learning rate for critics
    optim_alpha: float = 0.99  # RMSProp alpha
    optim_eps: float = 0.00001  # RMSProp epsilon
    grad_norm_clip: float = 10.0  # Reduce magnitude of gradients above this L2 norm


class AgentPrams(BaseModel):
    agent: str = "rnn"  # Default rnn agent
    rnn_hidden_dim: int = 64  # Size of hidden state for default rnn agent
    obs_agent_id: bool = True  # Include the agent's one_hot id in the observation
    obs_last_action: bool = True  # Include the agent's last action (one_hot) in the observation
    repeat_id: int = 1
    label: str = "default_label"


class Environment(BaseModel):
    name: str = 'Pogema-v0'
    grid_config: GridConfig = None
    integration: str = "PyMARL"


class AlgoSettings(BaseModel):
    QMIX: QMIX = QMIX()
    VDN: VDN = VDN()
    IQL: IQL = IQL()
    QTRAN: QTRAN = QTRAN()
    COMA: COMA = COMA()


class ExperimentConfig(BaseModel):
    env_args: Environment = Environment()
    agent_params: AgentPrams = AgentPrams()
    rl_params: RlParams = RlParams()
    logging: Logging = Logging()
    settings: PymarlSettings = PymarlSettings()
    algo_settings: AlgoSettings = AlgoSettings()
    algo: Literal["QMIX", "VDN", "IQL", "QTRAN"] = "QMIX"
