gym.register(
    id="Unitree-G1-29dof-Kick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kick_env_cfg:G1JuggleEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.kick_env_cfg:G1JuggleEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_kick_ppo_cfg:G1KickPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Kick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kick_env_cfg:G1JuggleEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.kick_env_cfg:G1JuggleEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_kick_ppo_cfg:G1KickPPORunnerCfg"
        ),
    },
)
