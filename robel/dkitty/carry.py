# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Carry tasks with DKitty robots.

This is a single movement from an initial position to a target position.
"""

import abc
import collections
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from robel.components.tracking import TrackerComponentBuilder, TrackerState
# from robel.dkitty.base_env import BaseDKittyUprightEnv
from robel.dkitty.walk import BaseDKittyWalk
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.math_utils import calculate_cosine
from robel.utils.resources import get_asset_path

DKITTY_ASSET_PATH = 'robel/dkitty/assets/dkitty_carry-v2.2.xml'

DEFAULT_OBSERVATION_KEYS = (
    'payload_error', 'payload_height'
)


class BaseDKittyCarry(BaseDKittyWalk, metaclass=abc.ABCMeta):
    """Shared logic for DKitty carry tasks."""

    def __init__(self,
                 asset_path: str = DKITTY_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 target_tracker_id: Optional[Union[str, int]] = None,
                 heading_tracker_id: Optional[Union[str, int]] = None,
                 payload_tracker_id: Optional[Union[str, int]] = None,
                 deliver_tracker_id: Optional[Union[str, int]] = None,
                 frame_skip: int = 40,
                 upright_threshold: float = 0.9,
                 upright_reward: float = 1,
                 falling_reward: float = -500,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            target_tracker_id: The device index or serial of the tracking device
                for the target location.
            heading_tracker_id: The device index or serial of the tracking
                device for the heading direction. This defaults to the target
                tracker.
            frame_skip: The number of simulation steps per environment step.
            upright_threshold: The threshold (in [0, 1]) above which the D'Kitty
                is considered to be upright. If the cosine similarity of the
                D'Kitty's z-axis with the global z-axis is below this threshold,
                the D'Kitty is considered to have fallen.
            upright_reward: The reward multiplier for uprightedness.
            falling_reward: The reward multipler for falling.
        """
        self._target_tracker_id = target_tracker_id
        self._heading_tracker_id = heading_tracker_id
        self._deliver_tracker_id = deliver_tracker_id
        self._payload_tracker_id = payload_tracker_id
        if self._heading_tracker_id is None:
            self._heading_tracker_id = self._target_tracker_id
        if self._deliver_tracker_id is None:
            self._deliver_tracker_id = self._target_tracker_id

        super().__init__(
            asset_path = DKITTY_ASSET_PATH)
        self._observation_keys += DEFAULT_OBSERVATION_KEYS # append to observations
        if self._payload_tracker_id is None:
            self._payload_tracker_id = self._torso_tracker_id

        self._initial_target_pos = np.zeros(3)
        self._initial_heading_pos = None
        self._initial_deliver_pos = None

    def _configure_tracker(self, builder: TrackerComponentBuilder):
        """Configures the tracker component."""
        super()._configure_tracker(builder)
        builder.add_tracker_group(
            'payload',
            vr_tracker_id=self._payload_tracker_id,
            sim_params=dict(
                element_name='payload',
                element_type='site',
            ),
            mimic_xy_only=True)
        builder.add_tracker_group(
            'deliver',
            vr_tracker_id=self._deliver_tracker_id,
            sim_params=dict(
                element_name='deliver',
                element_type='site',
            ),
            mimic_xy_only=True)

    def _reset(self):
        """Resets the environment."""
        self._reset_dkitty_standing()

        # If no heading is provided, head towards the target.
        target_pos = self._initial_target_pos
        heading_pos = self._initial_heading_pos
        deliver_pos = self._initial_deliver_pos
        if heading_pos is None:
            heading_pos = target_pos.copy()
            heading_pos[2] = 0.25
        if deliver_pos is None:
            deliver_pos = heading_pos.copy()

        # Set the tracker locations.
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot=np.identity(3)),
            'target': TrackerState(pos=target_pos),
            'heading': TrackerState(pos=heading_pos),
            'deliver': TrackerState(pos=deliver_pos),
        })

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        obs_dict = super().get_obs_dict()
        payload_state, deliver_state = self.tracker.get_state(['payload', 'deliver'])
        obs_dict.update({'payload_error': deliver_state.pos[:2] - payload_state.pos[:2]}) 
        obs_dict.update({'payload_height': payload_state.pos[2]}) 
        return obs_dict


    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        reward_dict = super().get_reward_dict(action, obs_dict)
        
        payload_xy_dist = np.linalg.norm(obs_dict['payload_error'])
        payload_drop = np.linalg.norm(0.4 - obs_dict['payload_height'])
        reward_dict.update({'payload_dist': -10*payload_xy_dist})
        reward_dict.update({'payload_drop': -10*payload_drop})
        reward_dict['bonus_small'] += 5 * (payload_xy_dist < 0.5)
        reward_dict['bonus_big'] *= (payload_xy_dist < 0.5)
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        return collections.OrderedDict((
            ('points', -np.linalg.norm(obs_dict['payload_error'])),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def get_done(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Returns whether the episode should terminate."""
        kitty_fall = obs_dict[self._upright_obs_key] < self._upright_threshold
        object_fall = obs_dict["payload_height"] < 0.100
        # return object_fall or kitty_fall
        return kitty_fall


@configurable(pickleable=True)
class DKittyCarryFixed(BaseDKittyCarry):
    """Carry straight towards a fixed location."""

    def _reset(self):
        """Resets the environment."""
        target_dist = 2.0
        target_theta = np.pi / 2  # Point towards y-axis
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])
        super()._reset()


@configurable(pickleable=True)
class DKittyCarryRandom(BaseDKittyCarry):
    """Carry straight towards a random location."""

    def __init__(
            self,
            *args,
            target_distance_range: Tuple[float, float] = (1.0, 2.0),
            # +/- 60deg
            target_angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
            **kwargs):
        """Initializes the environment.

        Args:
            target_distance_range: The range in which to sample the target
                distance.
            target_angle_range: The range in which to sample the angle between
                the initial D'Kitty heading and the target.
        """
        super().__init__(*args, **kwargs)
        self._target_distance_range = target_distance_range
        self._target_angle_range = target_angle_range

    def _reset(self):
        """Resets the environment."""
        target_dist = self.np_random.uniform(*self._target_distance_range)
        # Offset the angle by 90deg since D'Kitty looks towards +y-axis.
        target_theta = np.pi / 2 + self.np_random.uniform(
            *self._target_angle_range)
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])
        super()._reset()

# TODO: Should randomize payload as well
@configurable(pickleable=True)
class DKittyCarryRandomDynamics(DKittyCarryRandom):
    """Carry straight towards a random location."""

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()
