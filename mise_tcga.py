import constants as constants
from continuous.tcga.tcga_eval import compute_eval_metrics
from utils import data_utils as du
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm
from utils import common_utils as cu

logging.basicConfig(
    filename="tcga_out.log",
    format="%(asctime)s :: %(filename)s:%(funcName)s :: %(message)s",
    filemode="a+",
)
logger = logging.getLogger(name="continuous_ite")
logger.setLevel(logging.DEBUG)
constants.logger = logger

dataset_name = constants.TCGA_SINGLE_0
model_pattern = "factual/best_val_model-?.pt"
gpu_id = 1

cu.set_cuda_device(gpu_id)
cu.set_seed(0)

mises = []
for ds_seed in tqdm(range(10)):
    all_matrix, test_matrix, t_grid, indim, data_class, es = du.get_dataset_seed(
        dataset_name=dataset_name, dataset_num=ds_seed,
        num_epoch=100,  # dummy for now
        suffix="dummy",
        results_path="dummy"
    )
    test_dosages, test_xs, test_ys = constants.matrix_to_txy(
        test_matrix, cu.get_device()
    )

    results_path = Path(
        constants.fcf_dir / f'continuous/results/{dataset_name}/GI/Vcnet_tr')
    kwargs = {constants.LOAD_MODEL: model_pattern,
              constants.MODEL_TYPE: constants.VCNET_TR}

    model = du.load_model(
        dataset_name=dataset_name,
        dataset_num=ds_seed,
        indim=indim,
        results_path=results_path,
        **kwargs,
    )

    mise, _, _ = compute_eval_metrics(
        dataset_name=dataset_name,
        dataset=data_class.dataset,
        test_patients=test_xs,
        num_treatments=1,
        model=model,
        train=False,
    )

    mises.append(mise)

print(mises)
print(np.mean(mises))
print(np.std(mises))
