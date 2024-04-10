# Hide pygame support prompt
from gymnasium.envs.registration import register
# import os
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


def register_mapf_envs():
    """Import the envs module so that envs register themselves."""
    # base_env.pu
    register(
        id='mapf-v0',
        entry_point='mMAPF_env.MAPF_env.envs.baseenv:MAPFEnv',
    )
