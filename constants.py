from pathlib import Path
from torchvision import transforms
import torch
from logging import Logger

GENERAL_SPECS = "general_specs"
THETA_SPECS = "theta_specs"
CNF_SPECS = "cnf_specs"
KWARGS = "kwargs"
BASELINE_ARGS = "baseline_args"
GIKS_ARGS = "GIKS_args"
TR_LAMBDA = "target_reg_lambda_baselines"
GI_LAMBDA = "GI_lamda"
GP_LAMBA = "gp_lambda"
GP_NUM_EXPLORE = "num_samples_to_explore_for_GP"
BTM_K_VAR = "bottom_k_variance_terms"
GP_KERNEL = "Which_kernel_for_GP"
COSINE_KERNEL = "cosine"
RBF_KERNEL = "rbf_kernel"
DOTPRODUCT_KERNEL = "dot_product"
NTK_KERNEL = "ntk_kernel"
CTR_SAMPLING_DIST = "sample_counterfactuals_from_distribution"
FORCE_FAR_DOSAGES = "force_far_dosages"
FORCE_GP_SUPERVISION = "force_gp_supervision"

RETURN_VARIANCE = "return_variance"
UNIFORM = "unf_distribution"
INV_PROP = "inverse_propensity_distribution"
MARGINAL_T_DIST = "marginal_dosage_distribution"

FWD_FNC = "forward_fnc"
LOGVAL_EPOCHS = "log_valmetrics_every"


DATASET = "dataset"
DATASET_NUM = "dataset_number"
TEST_DATASET = "test_dataset"
MODEL_NAME = "model_name"

MODEL_TYPE = "model_type"
SYNTHETIC = "synthetic"
SYNTHETIC_CONT = "syn"
IHDP_CONT = "ihdp"
TCGA_SINGLE_0 = "tcga_single_0"
TCGA_SINGLE_1 = "tcga_single_1"
TCGA_SINGLE_2 = "tcga_single_2"
NEWS_CONT = "news"
SIMU1 = "toy_syn"

TRNDATA_SUBSET = "training_data_subset"
FULL_DATA = "full_data"
BUDGET = "budget"

LOGREG = "LR"
RESNET = "resnet"

INIT_THETA = "initialize_theta_from_scratch"
LOAD_MODEL = "load_model"
TEST_CHECKPOINTS = "Checkpoints_needed_at_epochs"
CHECKPOINTS = "Checkpoints_needed_at_epochs"
EVAL_CHECKPOINTS = "Evaluate_model_at_Checkpoints_needed_at_epochs"
START_EPOCHS = "model_start_epochs"

GNLL = "gnll"
SM = "softmax_filtering"
SM_TEMP = "softmax_filtering_temperature"
LOG_VAL_MSE = "log_val_mse"
GP_PARAMS_TUNING = "GP_params_tuning"
GI_DELTA_TUNING = "GI_delta_tuning"

TRAIN_ARGS = "train_args"

THETA = "th"
CNF = "cnf"
TUNE_THETA = "fine_tune_greedy_theta"

LRN_RATE = "lr"
MOMENTUM = "momentum"
OPTIMIZER = "optimizer"
BATCH_NORM = "batch_norm"
SAMPLER = "sampler"
SCRATCH = "scratch"
NNARCH = "nn_arch"
DROPOUT = "dropout"
IMGEMB_DROPOUT = "dropout_in_image_embedding"
IRM_W = "irm_w"
DANN_ALPHA = "dann_alpha"
PRETRN_EPOCHS = "epochs_of_pretraining"

ENFORCE_BASELINE = "enforce_baseline_losses"
GP_UNF_T = "sample_unf_t_in_GP_phase"
CTR_PER_X = "num_ctr_loss_perex"
RETURN_EMB = "return_embeddings_in_fwd"

SW = "summarywriter"
BATCH_SIZE = "batch_size"
EXPT_NAME = "experiment_name"
SEED = "seed"
VERBOSE = "verbose"
GPU_ID = "gpu_id"
PLOT_ADRF = "plot_dose_response_curve"
RUN_ALGOS = "run_algorithms"
LOGGER_PATH = "logger_path"
SAVE_MODEL = "save_model"
RESULTS_PATH = "pkl_path"
GI_LINEAR_DELTA = "gi_linear_delta"
GP_LINEAR_DELTA = "GP_ctr_linear_delta"
NUM_SAMPLES_LINDELTA = "number_of_samples_for_GI"
NEED_FAR_GP = "Do_we_need_beta-ctr_for_faraway_betas"
TRIGGER_FAR_CTR = "Trigger_training_of_far_beta"


ONLY_FAR_CTR = "run_only_GP_or_attn-No_GI"

TARNET = "Tarnet"
TARNET_TR = "Tarnet_tr"
DRNET = "Drnet"
DRNET_TR = "Drnet_tr"
VCNET = "Vcnet"
VCNET_GIGP = "Vcnet_gigp_ours"
VCNET_TR = "Vcnet_tr"
VCNET_HSIC = "Vcnet_hsic_regularizer"
TRANSTEE = "transtee"

HPM_TUNING = "HPM_tuning_expts"

SCHEDULER = "scheduler"
SCHEDULER_TYPE = "scheduler_type"
EPOCHS = "epochs"
TB_DIR = Path("tblogs/")
FILE_SUFFIX = "suffix_pickle_name"
LOG_DIR = Path("results/logs/")
THRESHOLD = "threshold"
INFTY = 1e7
BILEVEL_CTR_PREDS = "method_for_ctr_preds"

GNLL_LOSS: torch.nn.GaussianNLLLoss = torch.nn.GaussianNLLLoss(reduction="mean")
GNLL_LOSS_PEREX: torch.nn.GaussianNLLLoss = torch.nn.GaussianNLLLoss(reduction="none")

matrix_to_txy = lambda matrix, device: (
    matrix[:, 0].to(device, dtype=torch.float64),
    matrix[:, 1:-1].to(device, dtype=torch.float64),
    matrix[:, -1].to(device, dtype=torch.float64),
)

logger: Logger = None
fcf_dir = Path(".").absolute()
