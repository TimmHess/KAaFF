YAML_CONFIG=""
DSET_ROOTPATH=""
SAVE_PATH=""
EXP_NAME=""

SEEDS=(42 52 62 72 82)
EXCLUDE_TASKS=(1 5 10 15)

while test $# -gt 0; 
do
    case "$1" in
        --yaml_config)
            shift
            YAML_CONFIG=$1
            shift
            ;;
        --dset_rootpath)
            shift
            DSET_ROOTPATH=$1
            shift
            ;;
        --save_path)
            shift
            SAVE_PATH=$1
            shift
            ;;
        --exp_name)
            shift
            EXP_NAME=$1
            shift
            ;;
        *)
            echo "$1 is not a recognized flag! Use --yaml_config, --dset_rootpath, --save_path, --exp_name."
            exit 1;
            ;;
    esac
done  

echo ""
echo "yaml config : $YAML_CONFIG";
echo "dset rootpath : $DSET_ROOTPATH";
echo "result save path : $SAVE_PATH";
echo "experiment name : $EXP_NAME";

if [ "$EXP_NAME" == "" ]; 
then
    echo "--exp_name must be set"
    exit 1
fi
if [ "$YAML_CONFIG" == "" ]; then
    echo "--yaml_config must be set"
    exit 1
fi
if [ "$DSET_ROOTPATH" == "" ]; then
    echo "--dset_rootpath must be set"
    exit 1
fi
if [ "$SAVE_PATH" == "" ]; then
    echo "--save_path must be set"
    exit 1
fi

# Default Continual Learing Runs
for SEED in ${SEEDS[@]}
do 
python train.py --save_path $SAVE_PATH --dset_rootpath $DSET_ROOTPATH --num_experiences 10 --eval_max_iterations 0 --config_path $YAML_CONFIG --exp_name $EXP_NAME"-"$SEED --seed $SEED
done

# Runs of the Exclusion-Experiments
for EXC in ${EXCLUDE_TASKS[@]}
do 
    for SEED in ${SEEDS[@]}
    do 
    python train.py --save_path $SAVE_PATH --dset_rootpath $DSET_ROOTPATH --num_experiences 10 --eval_max_iterations 0 --config_path $YAML_CONFIG --exp_name $EXP_NAME"_"$EXC"-"$SEED --seed $SEED --exclude_experiences $EXC
    done
done