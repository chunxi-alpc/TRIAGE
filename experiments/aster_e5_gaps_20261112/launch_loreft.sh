set -e
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/aster-e5-gaps-20261112
python experiments/aster_e5_gaps_20261112/loreft_native.py --action controller --gpus 0,1
python experiments/aster_e5_gaps_20261112/finalize_loreft.py
