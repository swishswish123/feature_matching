#$ -l gpu=true
#$ -l h_rt=0:30:0
#$ -l tmem=10G
#$ -N frames
#$ -wd /home/aenkaoua/matching
#$ -S /bin/bash
#!/bin/bash
source feature_matching_env/bin/activate
source /share/apps/source_files/python/python-3.7.2.source
source /share/apps/source_files/cuda/cuda-10.1.source
$(command -v /share/apps/python-3.7.2-shared/bin/python3.7) endo_data_frames_from_vid.py