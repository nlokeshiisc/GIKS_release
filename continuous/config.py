import constants as constants
from pathlib import Path
import numpy as np

config = {
    constants.DATASET: constants.IHDP_CONT,
    constants.DATASET_NUM: np.arange(50),  # np.arange(10)
    # This is irrespective of fine tuning -- how many u want to run?
    constants.EPOCHS: 401,
    constants.ENFORCE_BASELINE: False,
    constants.BASELINE_ARGS: {
        constants.BATCH_SIZE: 128,
        constants.SAVE_MODEL: True,
        # %%
        # This is for the baselines
        constants.RUN_ALGOS: [constants.VCNET_TR],
        constants.TR_LAMBDA: 1,
        constants.CHECKPOINTS: [],  # epochs at whoch models wil be dumped
    },
    constants.GIKS_ARGS: {
        constants.GI_LAMBDA: 1e-1,
        constants.BATCH_NORM: False,
        constants.PRETRN_EPOCHS: 300,  # Set to a high value if only factual loss is needed
        constants.BATCH_SIZE: 128,  # set 32 for ihdp and 128 for tcga
        # important Should we sample the counterfactuals uniformly during the GP phase?
        constants.GP_UNF_T: True,
        constants.GI_LINEAR_DELTA: 0.05,
        constants.GP_LINEAR_DELTA: 0.1,
        constants.NUM_SAMPLES_LINDELTA: 10,
        constants.NEED_FAR_GP: True,  # important
        # important  This is a flag that overrides everything and runs just the GP or attn for the far away beta
        constants.ONLY_FAR_CTR: False,
        constants.TRIGGER_FAR_CTR: 300,  # important Same flag triggers attn too
        constants.HPM_TUNING: False,
        constants.MODEL_TYPE: constants.VCNET_TR,
        constants.GP_LAMBA: 1e-1,  # important
        # important If finetuning, this flas assumes as though training is resumed
        constants.START_EPOCHS: 300,
        # This flag is introduced as parts of the rebuttal question on which distribution to sample counterfactuals from?
        constants.CTR_SAMPLING_DIST: constants.UNIFORM,
        constants.LRN_RATE: 1e-2,  # important
        # "factual/ckpt-200-?.pt", # ? would be replaced by dataset num else give None
        constants.LOAD_MODEL: None,
        # important Give a number for bottom-k filtering. Else give GNLL/SM
        constants.BTM_K_VAR: constants.SM,
        constants.GP_KERNEL: constants.COSINE_KERNEL,
        constants.SM_TEMP: 0.5,
        constants.LOG_VAL_MSE: False,
        constants.GP_PARAMS_TUNING: False,
        constants.GI_DELTA_TUNING: False,
    },
    constants.SEED: 0,
    # important This will be appended to every file that gets saved
    constants.FILE_SUFFIX: "GI_model",
    constants.GPU_ID: 0,  # important
}

# %%
