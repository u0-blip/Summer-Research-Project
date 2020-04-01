conda activate pmp

if [ $1 = run_diff_dist ]; then
    for i in 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8
        do
            python main.py $i meep_out/prism_dis_$i.bin &
        done
else
    python main.py 0.5 meep_out/cube_dis_0.5_absorb.bin &
fi
wait
