#!/bin/bash

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
DATASETS="imagenet"
ATTACKS="fgsm bim std pgd df cw"
# ATTACKS="std"
DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
CLF="LR RF"
IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000"
NRSAMPLES="1500"

#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for net in $DATASETS; do
        if [ "$net" == imagenet ]; then
            python -u generate_clean_data.py --net "$net" --num_classes 1000
        fi 
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
attacks ()
{
    log_msg "Attack Clean Data with Foolbox Attacks and Autoattack!"

    for net in $DATASETS; do
        for att in $ATTACKS; do
            if [ "$net" == imagenet ]; then
                if [ "$att" == std ]; then
                    python -u attacks.py --net "$net" --attack "$att" --img_size 32 --batch_size 128 --num_classes 1000
                else
                    python -u attacks.py --net "$net" --attack "$att" --img_size 32 --batch_size 500 --num_classes 1000
                fi
            fi 

        done
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
extractcharacteristics ()
{
    log_msg "Extract Characteristics"

    for net in $DATASETS; do
        for att in $ATTACKS; do  
            for det in $DETECTORS; do
                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000
            done
        done
    done
}

# #-----------------------------------------------------------------------------------------------------------------------------------
detectadversarials ()
{
    log_msg "Detect Adversarials!"
    for net in $DATASETS; do
            for att in $ATTACKS; do
                for det in $DETECTORS; do
                    for nrsamples in $NRSAMPLES; do
                        for classifier in $CLF; do
                            python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000
                        done
                    done
                done
            done
        
    done
}



# for nr in {1,2,3,4}; do
#     echo "Generate Clean Data:  run: $nr" 
#     python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/clean_data', net=['imagenet'] )"
#     genereratecleandata
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/clean_data', net=['imagenet'], dest='./log_evaluation/imagenet', run_nr=$nr)"
# done


# for nr in {1,2,3,4}; do
#     echo "Attacks:  run: $nr" 
#     python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/attacks',    net=['imagenet'] )"
#     attacks
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',  net=['imagenet'], dest='./log_evaluation/imagenet', run_nr=$nr)"
# done


# for nr in 1; do
#     python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['imagenet'], dest='./log_evaluation/imagenet', run_nr=$nr)"
#     extractcharacteristics
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics', net=['imagenet'], dest='./log_evaluation/imagenet', run_nr=$nr)"
# done

# python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/clean_data', net=['imagenet'] )"
# python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/attacks',    net=['imagenet'] )"

# genereratecleandata
# attacks

# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/clean_data', net=['imagenet'], dest='./log_evaluation/imagenet', run_nr=2)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',    net=['imagenet'], dest='./log_evaluation/imagenet', run_nr=2)"

# extractcharacteristics
# detectadversarials

# #-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0

# : <<'END'
#   just a comment!
# END


# TODO List
# [] Run one time
#    [x] Generate Clean Data
#    [x] Generate Attacks
#    [] Generate Extract Charactersitics
#    [] Opimtize Params Charactersitics
#       [] Input MFS PFS
#       [] Layer MFS PFS
#       [] LID
#       [] Mahannobis
#       [] Statistical Test
#    [] Generate LR RF
# [] Save in file structure
# [] Create CSV

# [] Variance
#   [] Run 1
#   [] Run 2 
#   [] Run 3


# !!! 