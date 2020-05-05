# profile.py

import pstats, cProfile

import process_inp_file

process_inp_file.run()

# cProfile.runctx("process_inp_file.run()", globals(), locals(), "Profile.prof")

# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()
