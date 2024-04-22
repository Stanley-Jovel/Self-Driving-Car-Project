extends VehicleBody3D
class_name Car

@export var playing_area_x_size: float = 20
@export var playing_area_z_size: float = 20

@export var acceleration: float = 200
@export var max_steer_angle: float = 20

@export var track_path: Path3D

@onready var max_velocity = acceleration / mass * 40
@onready var ai_controller: CarAIController = $AIController3D
@onready var raycast_sensor: RayCastSensor3D = $RayCastSensor3D


var requested_acceleration: float
var requested_steering: float
var _initial_transform: Transform3D
var times_restarted: int
var delta_tracker: float = 0
#var last_position: Vector3

# Track Info
#var track_length: float
#var previous_offset: float
#var current_offset: float

var episode_ended_unsuccessfully_reward: float = -6

var _rear_lights: Array[MeshInstance3D]
@export var braking_material: StandardMaterial3D
@export var reversing_material: StandardMaterial3D

func get_normalized_velocity():
	return linear_velocity.normalized() * (linear_velocity.length() / max_velocity)
	
func _ready():
	ai_controller.init(self)
	_initial_transform = transform
	#track_length = track_path.curve.get_baked_length()
	#last_position = global_transform.origin
	_rear_lights.resize(2)
	_rear_lights[0] = $"car_base/Rear-light" as MeshInstance3D
	_rear_lights[1] = $"car_base/Rear-light_001" as MeshInstance3D

#func update_current_offset():
	#current_offset = track_path.curve.get_closest_offset(global_position)
	
func update_reward():
	var raycasts = raycast_sensor.get_observation()
	var left_sensor = raycasts[20]
	var center_sensor = raycasts[10]
	var right_sensor = raycasts[0]
	
	var desired_right_measure = 0.871
	var desired_center_measure = 0.8
	
	var right_sensor_reward = 0
	var center_sensor_reward = 0
	
	var reward = 0
	
	# Reward for not crashing head first
	if center_sensor <= desired_center_measure:
		var distance = abs(center_sensor - desired_center_measure)
		center_sensor_reward += lerp(1.0, 50.0, distance)
	else:
		center_sensor_reward = -50
	
	# Reward remain in the right lane
	var distance = 1 - abs(right_sensor - desired_right_measure)
	right_sensor_reward += lerp(1.0, 50.0, distance)
	
	# Penalization for deviating from the right lane
	if right_sensor < 0.844 or right_sensor > 0.896:
		#right_sensor_reward *= 0.15
		right_sensor_reward = -50 * 2
	
	if right_sensor < left_sensor:
		right_sensor_reward = -50 * 2
	
	reward += right_sensor_reward + center_sensor_reward
	
	# Penalization for moving backwards
	var velocity = get_normalized_velocity_in_player_reference().z
	if velocity > 0:
		reward += (velocity * 60) # ** 2
	else:
		reward -= 50
	
	ai_controller.reward = reward
	
	# Penalization for steering too much
	#var steering_penalty = abs(steering) * 2 
	#ai_controller.reward -= steering_penalty
	
func reset():
	times_restarted += 1
	
	transform = _initial_transform
	linear_velocity = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	
	
func _physics_process(delta):
	#update_reward()
	#update_current_offset()
	delta_tracker = delta
	if (ai_controller.heuristic != "human"):
		engine_force = (requested_acceleration) * acceleration
		steering = requested_steering
	else:
		ai_controller.set_action()

		
	_update_rear_lights()
	if (Input.is_action_just_released("finish_record_demo")):
		ai_controller.done = true
	#_reset_on_out_of_bounds()
	
	#if ai_controller.n_steps > ai_controller.reset_after:
		#ai_controller.n_steps = 0
		#reset()

func human_controls_car():
	var logal_engine_force = (
			int(Input.is_action_pressed("move_forward")) -
			int(Input.is_action_pressed("move_backward"))
		) #* acceleration
	var local_steering = lerp(steering, Input.get_axis("steer_right", "steer_left") * 0.40, 5 * delta_tracker)

	return [logal_engine_force, local_steering]

func _update_rear_lights():
	var velocity := get_normalized_velocity_in_player_reference().z
	
	set_rear_light_material(null)
	
	var brake_or_reverse_requested: bool
	if (ai_controller.heuristic != "human"):
		brake_or_reverse_requested = requested_acceleration < 0
	else:
		brake_or_reverse_requested = Input.is_action_pressed("move_backward")
	
	if velocity >= 0:
		if brake_or_reverse_requested:
			set_rear_light_material(braking_material)
	elif velocity <= -0.015:
		set_rear_light_material(reversing_material)

func set_rear_light_material(material: StandardMaterial3D):
	_rear_lights[0].set_surface_override_material(0, material)
	_rear_lights[1].set_surface_override_material(0, material)

func _end_episode(final_reward: float = 0.0):
	ai_controller.reward += final_reward
	ai_controller.needs_reset = true
	ai_controller.done = true

func _reset_if_needed():
	if ai_controller.needs_reset:
		reset()
		ai_controller.reset()
		
func get_normalized_velocity_in_player_reference() -> Vector3:
	return (
		global_transform.basis.inverse() *
		get_normalized_velocity()
		)
