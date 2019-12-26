
print("Testing the opti track client ====================================")
import opti_track.client as ot_client
ot = ot_client.OTClient()
poses = ot.get_poses()
print(poses)
ot.close()

# print("Testing the opti_tracker =========================================")
from opti_tracker import OTTrackerGroupConfig
from opti_tracker import OTTrackerComponent

from robel.simulation.sim_scene import SimScene
from robel.simulation.sim_scene import SimScene, SimBackend

sim = SimScene.create(model_handle = '/home/vik/Libraries/robel_dev/robel/dkitty/assets/dkitty_orient-v0.xml', backend=SimBackend.MUJOCO_PY)


from robel.components.tracking import TrackerComponentBuilder, TrackerState
tracker_builder = TrackerComponentBuilder()
configure_tracker(tracker_builder)
builder.add_tracker_group(
            'target',
            vr_tracker_id=self._target_tracker_id,
            sim_params=dict(
                element_name='target',
                element_type='site',
            ),
            mimic_xy_only=True)

# tracker_builder.set_state({
#             'torso': TrackerState(
#                 pos=np.zeros(3),
#                 rot_euler=np.array([0, 0, self._initial_angle])),
#             'target': TrackerState(
#                 pos=np.array([
#                     # The D'Kitty is offset to face the y-axis.
#                     np.cos(self._target_angle + np.pi / 2),
#                     np.sin(self._target_angle + np.pi / 2),
#                     0,
#                 ])),

gc = OTTrackerGroupConfig(sim_scene = sim )