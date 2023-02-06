export CUDA_VISIBLE_DEVICES=7
D_ROOT_DIR="attack-log/"
DETECTOR=ue
MODEL=("bert") #generic name for models; Options: ("bert", "roberta")
DATASET="sst2"   #Options: ("imdb" , "ag-news", "sst2")
MODEL_DATASET="SST-2" #Change to "SST-2" for "sst2" only
TARGET_MODEL=("textattack/bert-base-uncased-$MODEL_DATASET")

RECIPE="textfooler textfooler_high_confidence_0.9 bae bae_high_confidence_0.9 pruthi pruthi_high_confidence_0.9 textbugger textbugger_high_confidence_0.9" #Four attack options (No tf-adj for sst2 dataset)
EXP_NAME="tmp" #name for experiment
PARAM_PATH="params/reduce_dim_100.json" #Indicate model parameters (e.g. No PCA, linear PCA, MLE)
SCEN="s1"  #Scenario (see paper for details); Options: ("s1" "s2")
ESTIM="MCD"  #Options : ("None", "MCD")

START_SEED=0
END_SEED=0
GPU=0

for ((i=0; i< ${#MODEL[@]}; i++ ));
do
  for recipe in $RECIPE
  do
    python main.py --detect_method $DETECTOR --dataset $DATASET --model_type ${MODEL[i]}\
    --attack_type ${recipe} --scenario $SCEN --cov_estimator $ESTIM\
    --start_seed $START_SEED --end_seed $END_SEED --model_params_path $PARAM_PATH\
    --exp_name $EXP_NAME --gpu $GPU --target_model ${TARGET_MODEL[i]} --data_root_dir $D_ROOT_DIR
  done
done
