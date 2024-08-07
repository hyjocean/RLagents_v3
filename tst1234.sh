python scripts_0.py --gpu_id=0 --map_size=20 --map_density=0.01 --imi_prob=0.2 --wandb_log --render_mode &
p3=$! 
echo "Process 1 PID: $p3" 
python scripts_0.py --gpu_id=0 --map_size=20 --map_density=0.1 --imi_prob=0.2 --wandb_log  --render_mode &
p4=$! 
echo "Process 1 PID: $p4" 
python scripts_0.py --gpu_id=0 --map_size=10 --map_density=0.01 --imi_prob=0.2 --wandb_log --render_mode &
p5=$!
echo "Process 1 PID: $p5"
wait $p3 $p4 $p5