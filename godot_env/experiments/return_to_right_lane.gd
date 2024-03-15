extends Node3D

@onready var car = $Car
@onready var road = $Track

var streets

# Called when the node enters the scene tree for the first time.
func _ready():
	streets = get_all_children(road)
	#self.reset_position()

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	if Input.is_action_just_released("restart_experiment"):
		car.ai_controller.done = true
		self.reset_position()
		
func get_all_children(in_node,arr:=[]):
	arr.push_back(in_node)
	for child in in_node.get_children():
		arr = get_all_children(child,arr)
	return arr
	
func reset_position():
	var random_index = randi() % streets.size()
	var street = streets[random_index]
	
	car.translate(street.position + Vector3(randf_range(-1.5, 1.5), 0, 0)) # far right
	#car.translate(street.position + Vector3(randf_range(0.6, 2), 0, 0)) # far left
	car.rotate_y(randi_range(0, 360))
	#car.acceleration = 0
	#car.steering = 0
