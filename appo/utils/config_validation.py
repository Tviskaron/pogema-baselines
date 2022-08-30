import multiprocessing
from typing import Union, Literal

from pogema import GridConfig
from pydantic import Extra, BaseModel, validator
import os
from os.path import join
from pogema.grid_config import Easy8x8, Normal8x8, Hard8x8, ExtraHard8x8
from pogema.grid_config import Easy16x16, Normal16x16, Hard16x16, ExtraHard16x16
from pogema.grid_config import Easy32x32, Normal32x32, Hard32x32, ExtraHard32x32
from pogema.grid_config import Easy64x64, Normal64x64, Hard64x64, ExtraHard64x64


class AsyncPPO(BaseModel, extra=Extra.forbid):
    experiment_summaries_interval: int = 20
    adam_eps: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    gae_lambda: float = 0.95
    rollout: int = 32
    num_workers: int = multiprocessing.cpu_count()
    recurrence: int = 32
    use_rnn: bool = True
    rnn_type: str = 'gru'
    rnn_num_layers: int = 1
    ppo_clip_ratio: float = 0.1
    ppo_clip_value: float = 1.0
    batch_size: int = 1024
    num_batches_per_iteration: int = 1
    ppo_epochs: int = 1
    num_minibatches_to_accumulate: int = -1
    max_grad_norm: float = 4.0

    exploration_loss_coeff: float = 0.003
    value_loss_coeff: float = 0.5
    kl_loss_coeff: float = 0.0
    exploration_loss: str = 'entropy'
    num_envs_per_worker: int = 2
    worker_num_splits: int = 2
    num_policies: int = 1
    policy_workers_per_policy: int = 1
    max_policy_lag: int = 10000
    traj_buffers_excess_ratio: int = 2
    decorrelate_experience_max_seconds: int = 10
    decorrelate_envs_on_one_worker: bool = True

    with_vtrace: bool = True
    vtrace_rho: float = 1.0
    vtrace_c: float = 1.0
    set_workers_cpu_affinity: bool = True
    force_envs_single_thread: bool = True
    reset_timeout_seconds: int = 120
    default_niceness: int = 0
    train_in_background_thread: bool = True
    learner_main_loop_num_cores: int = 1
    actor_worker_gpus = []

    with_pbt: bool = False
    pbt_mix_policies_in_one_env: bool = True
    pbt_period_env_steps: int = 5e6
    pbt_start_mutation: int = 2e7
    pbt_replace_fraction: float = 0.3
    pbt_mutation_rate: float = 0.15
    pbt_replace_reward_gap: float = 0.1
    pbt_replace_reward_gap_absolute: float = 1e-6
    pbt_optimize_batch_size: bool = False
    pbt_target_objective: str = 'true_reward'

    use_cpc: bool = False
    cpc_forward_steps: int = 8
    cpc_time_subsample: int = 6
    cpc_forward_subsample: int = 2
    benchmark: bool = False
    sampler_only: bool = False


class ExperimentSettings(BaseModel, extra=Extra.forbid):
    save_every_sec: int = 120
    keep_checkpoints: int = 1
    save_milestones_sec: int = -1
    stats_avg: int = 100
    learning_rate: float = 1e-4
    train_for_env_steps: int = 1e10
    train_for_seconds: int = 1e10

    obs_subtract_mean: float = 0.0
    obs_scale: float = 1.0

    gamma: float = 0.99
    reward_scale: float = 1.0
    reward_clip: float = 10.0

    encoder_type: str = 'conv'
    encoder_custom: str = 'pogema_residual'
    encoder_subtype: str = 'convnet_simple'
    encoder_extra_fc_layers: int = 1

    pogema_encoder_num_filters: int = 64
    pogema_encoder_num_res_blocks: int = 3

    hidden_size: int = 512
    nonlinearity: str = 'relu'
    policy_initialization: str = 'orthogonal'
    policy_init_gain: float = 1.0
    actor_critic_share_weights: bool = True

    use_spectral_norm: bool = False
    adaptive_stddev: bool = True
    initial_stddev: float = 1.0


class GlobalSettings(BaseModel, extra=Extra.forbid):
    algo: str = 'APPO'
    env: str = None
    experiment: str = None
    experiments_root: str = None
    train_dir: str = 'train_dir/experiment'
    device: str = 'gpu'
    seed: int = None
    cli_args: dict = {}
    use_wandb: bool = True


class Evaluation(BaseModel, extra=Extra.forbid):
    fps: int = 0
    render_action_repeat: int = None
    no_render: bool = True
    policy_index: int = 0
    record_to: str = join(os.getcwd(), '..', 'recs')
    continuous_actions_sample: bool = True
    env_frameskip: int = None


class Environment(BaseModel, ):
    grid_config: Union[GridConfig, Easy8x8, Normal8x8, Hard8x8, ExtraHard8x8,
                       Easy16x16, Normal16x16, Hard16x16, ExtraHard16x16,
                       Easy32x32, Normal32x32, Hard32x32, ExtraHard32x32,
                       Easy64x64, Normal64x64, Hard64x64, ExtraHard64x64] = GridConfig()
    name: Literal['pogema_v0'] = 'pogema'
    # framestack: int = 1
    max_episode_steps: int = 256
    animation_monitor: bool = False
    animation_dir: str = './renders'
    evaluation: bool = False
    grid_memory_radius: int = None
    path_to_grid_configs: str = None
    auto_reset: bool = True


class Experiment(BaseModel):
    name: str = None
    environment: Environment = Environment()
    async_ppo: AsyncPPO = AsyncPPO()
    experiment_settings: ExperimentSettings = ExperimentSettings()
    global_settings: GlobalSettings = GlobalSettings()
    evaluation: Evaluation = Evaluation()

    @validator('global_settings')
    def seed_initialization(cls, v, values):
        # if v.env is None:
        #     v.env = values['environment'].name
        if v.experiment is None:
            v.experiment = values['name']
        return v
