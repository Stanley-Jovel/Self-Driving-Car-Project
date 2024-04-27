extends AIController3D
class_name CarAIController

func get_obs() -> Dictionary:
	# Positions and velocities are converted to the player's frame of reference
	var player_velocity = _player.get_normalized_velocity_in_player_reference()

	var observations : Array = [
		player_velocity.x,
		player_velocity.z,
		_player.angular_velocity.y * 1.5,
		_player.steering / deg_to_rad(_player.max_steer_angle)
	]

	observations.append_array(_player.raycast_sensor.get_observation())
	return {"obs": observations}
	
## Returns the action that is currently applied to the robot.
func get_action():
	return [_player.requested_acceleration, _player.requested_steering]
	
func get_reward() -> float:
	var reward = _player.update_reward()
	zero_reward()
	return reward

func get_action_space() -> Dictionary:
	return {
		"acceleration" : {
			"size": 1,
			"action_type": "continuous"
		},
		"steering" : {
			"size": 1,
			"action_type": "continuous"
		},
	}

func set_action(action = null) -> void:
	if action:
		_player.requested_acceleration = clampf(action.acceleration[0], -1.0, 1.0)
		_player.requested_steering = clampf(action.steering[0], -1.0, 1.0)
	else:
		var r = _player.human_controls_car()
		_player.requested_acceleration = clampf(r[0], -1.0, 1.0)
		_player.requested_steering = clampf(r[1], -1.0, 1.0)

func reset():
	n_steps = 0
	needs_reset = false
	_player.reset()
