extends VehicleBody3D
class_name Car

@export var playing_area_x_size: float = 20
@export var playing_area_z_size: float = 20

@export var acceleration: float = 200
@export var max_steer_angle: float = 20

@onready var max_velocity = acceleration / mass * 40
@onready var ai_controller: CarAIController = $AIController3D
@onready var raycast_sensor: RayCastSensor3D = $RayCastSensor3D


var requested_acceleration: float
var requested_steering: float
var _initial_transform: Transform3D
var times_restarted: int
var delta_tracker: float = 0

var episode_ended_unsuccessfully_reward: float = -6

var _rear_lights: Array[MeshInstance3D]
@export var braking_material: StandardMaterial3D
@export var reversing_material: StandardMaterial3D

func get_normalized_velocity():
	return linear_velocity.normalized() * (linear_velocity.length() / max_velocity)
	
func _ready():
	ai_controller.init(self)
	_initial_transform = transform
	
	_rear_lights.resize(2)
	_rear_lights[0] = $"car_base/Rear-light" as MeshInstance3D
	_rear_lights[1] = $"car_base/Rear-light_001" as MeshInstance3D
	
func reset():
	times_restarted += 1
	
	transform = _initial_transform
	
	if randi_range(0, 1) == 0:
		transform.origin = -transform.origin
		transform.basis = transform.basis.rotated(Vector3.UP, PI)

	linear_velocity = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	
	transform.basis = transform.basis.rotated(Vector3.UP, randf_range(-0.3, 0.3))
	
func _physics_process(delta):
	#_update_reward()
	delta_tracker = delta

	if (ai_controller.heuristic != "human"):
		engine_force = (requested_acceleration) * acceleration
		steering = requested_steering
	else:
		var r = self.human_controls_car()
		ai_controller.set_action()

		
	_update_rear_lights()
	if (Input.is_action_just_released("finish_record_demo")):
		ai_controller.done = true
	#_reset_on_out_of_bounds()

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

#func _reset_on_out_of_bounds():
	#if (position.y < -2 or abs(position.x) > 10 or abs(position.z) > 10):
		#_end_episode(episode_ended_unsuccessfully_reward)

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
