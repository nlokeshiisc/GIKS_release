import os
import constants as constants
import numpy as np
from pathlib import Path
import utils.torch_utils as tu

import argparse

# import psutil
# per_cpu_pc = psutil.cpu_percent(interval=1, percpu=True) # get CPU usage
# num_cores = 4 # number of cores to use
# cores = np.argsort(per_cpu_pc)[:4] # get the indices of the lowest CPU usage
# pid = psutil.Process()  # get the process id
# print("CPU cores used: ", cores)
# pid.cpu_affinity(list(cores))   # Set the process to use the CPU with lowest usage

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, default=0, help="which partition of seed id")
parser.add_argument(
    "--num_parts", type=int, default=1, help="total number of partitions"
)
parser.add_argument("--ds", type=str, default=None, help="Which dataset?")
parser.add_argument("--gpu", type=int, default=None, help="which gpu to use")
args = parser.parse_args()

lrs = {
    constants.IHDP_CONT: 1e-2,
    constants.NEWS_CONT: 1e-3,
    constants.TCGA_SINGLE_0: 1e-4,
    constants.TCGA_SINGLE_1: 1e-4,
    constants.TCGA_SINGLE_2: 1e-4,
}

lambda_gis = {
    constants.IHDP_CONT: 1e-4,
    constants.NEWS_CONT: 1e-2,
    constants.TCGA_SINGLE_0: 1e-1,
    constants.TCGA_SINGLE_1: 1e-1,
    constants.TCGA_SINGLE_2: 1e-1,
}

lambda_gps = {
    constants.IHDP_CONT: 1e-1,
    constants.NEWS_CONT: 1e-4,
    constants.TCGA_SINGLE_0: 1e-2,
    constants.TCGA_SINGLE_1: 1e-2,
    constants.TCGA_SINGLE_2: 1e-2,
}

# This parameter has a one-to-one correspondence with the \sigma parameter of the GP
# We use softmax temperature for convenience
sm_temps = {
    constants.IHDP_CONT: 1,
    constants.NEWS_CONT: 0.05,
    constants.TCGA_SINGLE_0: 0.005,
    constants.TCGA_SINGLE_1: 0.005,
    constants.TCGA_SINGLE_2: 0.005,
}


def list_to_str(x):
    return str(list(x)).replace(" ", "")


all_seeds = {
    constants.IHDP_CONT: np.arange(50),
    constants.NEWS_CONT: np.arange(20),
    constants.TCGA_SINGLE_0: np.arange(10),
    constants.TCGA_SINGLE_1: np.arange(10),
    constants.TCGA_SINGLE_2: np.arange(10),
}


def resolve_ds(ds):
    if ds == "ihdp":
        return [constants.IHDP_CONT]
    elif ds == "news":
        return [constants.NEWS_CONT]
    elif ds in ["tcga_0", "t0", constants.TCGA_SINGLE_0]:
        return [constants.TCGA_SINGLE_0]
    elif ds in ["tcga_1", "t1", constants.TCGA_SINGLE_1]:
        return [constants.TCGA_SINGLE_1]
    elif ds in ["tcga_2", "t2", constants.TCGA_SINGLE_2]:
        return [constants.TCGA_SINGLE_2]


if args.gpu is not None:
    gpu_id = args.gpu
else:
    gpu_id = tu.get_available_gpus()

print(f"Scheduling the job in GPU: {gpu_id}")

part = args.part  # Are we parallelizing over seeds?
num_parts = args.num_parts  # How many parts are we parallelizing over?

if part is not None:
    assert num_parts is not None
    assert part < num_parts


datasets = [constants.IHDP_CONT]
if args.ds is not None:
    datasets = resolve_ds(args.ds)

for ds in datasets:
    lr = lrs[ds]
    ds_seeds = list_to_str(all_seeds[ds])
    lmda_gi = lambda_gis[ds]

    num_epochs = 101

    load_model = "factual-model/best_val_model-?.pt"

    expt = "gi-model"
    if num_parts > 1:
        expt = f"{expt}-{part}"
        ds_seeds = list_to_str(np.array_split(all_seeds[ds], num_parts)[part])

    ds_path = Path(constants.fcf_dir / f"continuous/results/{ds}/out")
    ds_path.mkdir(parents=True, exist_ok=True)

    command = f"python main_continuous.py\
                {constants.DATASET}:{ds}\
                {constants.DATASET_NUM}:{ds_seeds}\
                {constants.EPOCHS}:{num_epochs}\
                {constants.ENFORCE_BASELINE}:False\
                {constants.ENFORCE_GIKS}:True\
                {constants.GIKS_ARGS}.{constants.GI_LAMBDA}:{lambda_gis[ds]}\
                {constants.GIKS_ARGS}.{constants.PRETRN_EPOCHS}:200\
                {constants.GIKS_ARGS}.{constants.BATCH_SIZE}:128\
                {constants.GIKS_ARGS}.{constants.TRIGGER_FAR_CTR}:200\
                {constants.GIKS_ARGS}.{constants.START_EPOCHS}:200\
                {constants.GIKS_ARGS}.{constants.HPM_TUNING}:{False}\
                {constants.GIKS_ARGS}.{constants.LRN_RATE}:{lr}\
                {constants.GIKS_ARGS}.{constants.GI_LINEAR_DELTA}:0.05\
                {constants.GIKS_ARGS}.{constants.GP_LINEAR_DELTA}:0.05\
                {constants.GIKS_ARGS}.{constants.NEED_FAR_GP}:{False}\
                {constants.GIKS_ARGS}.{constants.TRIGGER_FAR_CTR}:200\
                {constants.GIKS_ARGS}.{constants.FAR_CTR_LAMBA}:{lambda_gps[ds]}\
                {constants.GIKS_ARGS}.{constants.LOAD_MODEL}:{load_model}\
                {constants.GIKS_ARGS}.{constants.LOG_VAL_MSE}:{False}\
                {constants.GIKS_ARGS}.{constants.GP_PARAMS_TUNING}:{False}\
                {constants.FILE_SUFFIX}:{expt}\
                {constants.GIKS_ARGS}.{constants.GP_KERNEL}:{constants.COSINE_KERNEL}\
                {constants.GIKS_ARGS}.{constants.BTM_K_VAR}:{constants.SM}\
                {constants.GIKS_ARGS}.{constants.SM_TEMP}:{sm_temps[ds]}\
                {constants.GPU_ID}:{gpu_id}"

    print(command)
    os.system(f"{command} > {str(ds_path.absolute())}/{expt}.out")
