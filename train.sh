#python main_benchmark.py --arg_file args/run_humanoid3d_walk_args.txt
#python main_benchmark.py --arg_file args/run_amp_humanoid3d_cartwheel_args.txt
#python main_benchmark.py --arg_file args/run_amp_dog3d_trot_args.txt
python mpi_run.py --arg_file args/run_amp_humanoid3d_run_args.txt --num_workers 4
