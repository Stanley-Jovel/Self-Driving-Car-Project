python sb3_imitation.py \
--env_path="./godot_env/Self_Driving_Car_Environment.app" \
--n_parallel=5 \
--il_timesteps=200_000 \
--rl_timesteps=500_000 \
--eval_episode_count=20 \
--speedup=15 \
--demo_files "./demos/circle_clock.json" "./demos/circle_counterclock.json" "./demos/eight.json" "./demos/snake.json" "./demos/snake_2.json"