import pickle as pkl
import argparse
from pathlib import Path
import continuous.main_helper as main_helper
import utils.common_utils as cu
import importlib.util
import logging
import constants as constants
import os
import continuous.config as config

config = config.config

this_dir = Path(".").absolute()
os.environ["QT_QPA_PLATFORM"] = "offscreen"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", help="Use the correct argument", default="continuous/config.py"
)
args, unknown_args = parser.parse_known_args()
file_name = os.path.splitext(args.config)[0]
file_name = file_name.split("/")[0]
spec = importlib.util.spec_from_file_location(file_name, args.config)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config


def evaluate_value(value):
    if "[" in value:
        # processing list arguments
        values = value.replace("[", "").replace("]", "")
        values = values.split(",")
        try:
            values = [eval(entry) for entry in values]
        except:
            pass
        return values
    else:
        try:
            return eval(value)
        except:
            return value


for override_config in unknown_args:
    parts = override_config.split(":")
    key = parts[0]
    value = parts[1]

    if "." in key:
        key_parts = key.split(".")
        primary_key = key_parts[0]
        secondary_key = key_parts[1]
        config[primary_key][secondary_key] = evaluate_value(value)
    else:
        config[key] = evaluate_value(value)
cu.dict_print(config)

gpuid = config[constants.GPU_ID]
cu.set_cuda_device(gpuid)
cu.set_seed(config[constants.SEED])


if __name__ == "__main__":
    num_epoch = config[constants.EPOCHS]
    dataset_name = config[constants.DATASET]
    dataset_nums = config[constants.DATASET_NUM]

    enforce_baseline = config[constants.ENFORCE_BASELINE]

    file_suffix = config[constants.FILE_SUFFIX]

    baseline_args = config[constants.BASELINE_ARGS]
    gi_args = config[constants.GIKS_ARGS]

    cu.set_dump_path(config=config)
    save_dir = cu.get_dump_path()

    (save_dir / "logs").mkdir(exist_ok=True, parents=True)
    (save_dir / "models").mkdir(exist_ok=True, parents=True)
    (save_dir / "pkl").mkdir(exist_ok=True, parents=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=str((save_dir / "logs" / f"{file_suffix}.log").absolute()),
        format="%(asctime)s :: %(filename)s:%(funcName)s :: %(message)s",
        filemode="w",
    )
    logger = logging.getLogger(name="continuous_ite")
    logger.setLevel(logging.DEBUG)
    logger.info(f"Running dataset: {dataset_name} for numbers: {dataset_nums}")
    logger.info(cu.dict_print(config))

    constants.logger = logger

    if enforce_baseline == False:
        # run GIKS
        gi_args = cu.insert_kwargs(gi_args, {constants.RESULTS_PATH: save_dir})
        assert gi_args[constants.NEED_FAR_GP] <= 1
        main_helper.run_GIKS(
            dataset_name=dataset_name,
            dataset_nums=dataset_nums,
            num_epoch=num_epoch,
            suffix=file_suffix,
            logger=logger,
            **gi_args,
        )

    else:
        baseline_args = baseline_args  # both ctr code and factual code is shared
        baseline_args = cu.insert_kwargs(
            baseline_args, {constants.RESULTS_PATH: save_dir}
        )
        main_helper.run_baselines(
            dataset_name=dataset_name,
            dataset_nums=dataset_nums,
            num_epoch=num_epoch,
            suffix=file_suffix,
            logger=logger,
            **baseline_args,
        )
