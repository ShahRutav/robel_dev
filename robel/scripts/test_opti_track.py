import robel
from mjrl.utils.gym_env import GymEnv
from robel.dkitty.orient import DKittyOrientRandom
from transforms3d.euler import euler2quat

e = DKittyOrientRandom(torso_tracker_id=0)
e._reset()
# e.viewer.setup()

while True:
	obs = e.get_obs_dict()
	torso_pos = obs['root_pos']
	torso_euler = obs['root_euler']
	torso_quat = euler2quat(0, 0, torso_euler[2], axes='rxyz')
	qp = e.sim.data.qpos.ravel().copy()
	qv = e.sim.data.qvel.ravel().copy()
	qp[:3] = torso_pos
	qp[3:6] = torso_euler
	for i in range(qp.shape[0]):
		e.sim.data.qpos[i] = qp[i]
	for i in range(qv.shape[0]):
		e.sim.data.qvel[i] = qv[i]
	# e.set_state(qp, qv)
	e.sim.forward()
	e.render()