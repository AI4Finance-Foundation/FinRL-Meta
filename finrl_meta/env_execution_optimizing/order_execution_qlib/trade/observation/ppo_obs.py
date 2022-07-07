import json
import math

import numpy as np
import pandas as pd
from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
from gym.spaces import Tuple

from .obs_rule import RuleObs


class PPOObs(RuleObs):
    """The observation defined in IJCAI 2020. The action of previous state is included in private state"""

    def get_obs(
        self,
        raw_df,
        feature_dfs,
        t,
        interval,
        position,
        target,
        is_buy,
        max_step_num,
        interval_num,
        action=0,
    ):
        if t == -1:
            self.private_states = []

        public_state = self.get_feature_res(feature_dfs, t, interval, whole_day=True)
        # market_state = feature_dfs[0].reshape(-1)[:6*240]
        private_state = np.array([position / target, (t + 1) / max_step_num, action])
        self.private_states.append(private_state)
        list_private_state = np.concatenate(self.private_states)
        list_private_state = np.concatenate(
            (
                list_private_state,
                [0.0] * 3 * (interval_num + 1 - len(self.private_states)),
            )
        )
        seqlen = np.array([interval])
        return np.concatenate((public_state, list_private_state, seqlen))
