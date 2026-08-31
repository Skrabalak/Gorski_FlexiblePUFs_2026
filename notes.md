steps for the merger
1. call inefficient unique combinations
1. translation from number letter pair to array of numbers
1. find best distance

find the best combination: This function uses a brute-force approach to find the best combination of particles, but multithreads it to greatly speed up the process.
    - rips through every possible combination
        - at each one it will calculate the distance and store the best found distance in that thread's storage locker
    - look through the storage locker and get the best (min function?)



    mkdir: cannot create directory ‘Lab/Experiment_2/Sarah’: No such file or directory
mkdir: cannot create directory ‘Code/Coding’: No such file or directory
mkdir: cannot create directory ‘Images/red_square_particle_details_images/’: No such file or directory
touch: cannot touch 'Lab/Experiment_2/Sarah': No such file or directory
touch: cannot touch 'Code/Coding': No such file or directory
touch: cannot touch 'Images/red_square_particle_details_images/data.csv': No such file or directory
PROCESSING_PARTICLE
Traceback (most recent call last):
  File "/N/project/Skrabalak/terame/Skrabalak Lab/Experiment_2/Sarah Altered Code/handle_prominent_features_combined.py", line 1605, in <module>
    save_particles("red_square_particle_details", red_square_particle_details)
  File "/N/project/Skrabalak/terame/Skrabalak Lab/Experiment_2/Sarah Altered Code/handle_prominent_features_combined.py", line 1121, in save_particles
    num_images = len(os.listdir(save_dir))
                     ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/N/project/Skrabalak/terame/Skrabalak Lab/Experiment_2/Sarah Altered Code/Coding Images/red_square_particle_details_images/'