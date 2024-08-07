python scripts_0.py --gpu_id=1 --map_size=40 --map_density=0.01  --imi_prob=0.2 --wandb_log --render_mode & 
p1=$! 
echo "Process 1 PID: $p1" 
python scripts_0.py --gpu_id=1 --map_size=40 --map_density=0.1   --imi_prob=0.2 --wandb_log --render_mode &
p2=$! 
echo "Process 2 PID: $p2" 
python scripts_0.py --gpu_id=1 --map_size=100 --map_density=0.01 --imi_prob=0.2 --wandb_log --render_mode &
p4=$! 
echo "Process 4 PID: $p4" 
python scripts_0.py --gpu_id=1 --map_size=100 --map_density=0.1  --imi_prob=0.2 --wandb_log --render_mode &
p5=$! 
echo "Process 5 PID: $p5" 
wait $p1 $p2 $p4 $p5