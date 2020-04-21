conda activate pmp
conda activate pmp
if [ $1 = run_diff_dist ]; then
    for i in 0.2 0.3 0.4 0.5 0.6 0.7
        do
            python main.py --distance $i --size 0.4 --visual 0 --file meep_out/new_dist/sphere2D_dist_s0.4_$i.bin &
        done
else
    python main.py --distance 0.5 --size 0.4 --visual 0 --file meep_out/cube_num_2_2d.bin
fi
wait
