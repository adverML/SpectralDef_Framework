#!/bin/bash

#  bash main_celebahq.sh &> log_evaluation/celebAHQ/all.log

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
# DATASETS="celebaHQ32 celebaHQ64 celebaHQ128"
# DATASETS="celebaHQ128"
# DATASETS="celebaHQ32"
DATASETS="celebaHQ256"

NUMCLASSES=4
VERSION="standard_4"
ATTACKS="cw"
# ATTACKS="fgsm bim pgd std df"

# ATTACKS="std"

# ATTACKS="std"

DETECTORS="InputMFS"
# DETECTORS="InputMFS LayerMFS"


# EPSILONS="4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."


CLF="LR RF"
IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000"
NRSAMPLES="1500"
NRRUN=1..4

#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for net in $DATASETS; do
        if [ "$net" == celebaHQ32 ]; then
            python -u generate_clean_data.py --net "$net" --num_classes $NUMCLASSES  --img_size 32
        fi 

        if [ "$net" == celebaHQ64 ]; then
            python -u generate_clean_data.py --net "$net" --num_classes $NUMCLASSES  --img_size 64
        fi 

        if [ "$net" == celebaHQ128 ]; then
            python -u generate_clean_data.py --net "$net" --num_classes $NUMCLASSES  --img_size 128
        fi 
        if [ "$net" == celebaHQ256 ]; then
            python -u generate_clean_data.py --net "$net" --num_classes $NUMCLASSES  --img_size 256
        fi 
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
attacks ()
{
    log_msg "Attack Clean Data with Foolbox Attacks and Autoattack!"

    for net in $DATASETS; do

        for att in $ATTACKS; do

            for eps in $EPSILONS; do

                if [ "$net" == celebaHQ32 ]; then
                    python -u attacks.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --img_size 32 --batch_size 500  --eps "$eps" --version "$VERSION"
                fi 

                if [ "$net" == celebaHQ64 ]; then
                    python -u attacks.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --img_size 64 --batch_size 500  --eps "$eps" --version "$VERSION"
                fi 

                if [ "$net" == celebaHQ128 ]; then
                    python -u attacks.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --img_size 128 --batch_size 64  --eps "$eps" --version "$VERSION"
                fi 


                if [ "$net" == celebaHQ256 ]; then
                    python -u attacks.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --img_size 256 --batch_size 12  --eps "$eps" --version "$VERSION"
                fi 
            done
        done
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
extractcharacteristics ()
{
    log_msg "Extract Characteristics"

    for net in $DATASETS; do
        for att in $ATTACKS; do  
            for eps in $EPSILONS; do
                for det in $DETECTORS; do
                
                    if [ "$net" == celebaHQ32 ]; then
                        python -u extract_characteristics.py --net "$net"  --num_classes $NUMCLASSES --attack "$att" --detector "$det" --eps "$eps"
                    fi 
                    if [ "$net" == celebaHQ64 ]; then
                        python -u extract_characteristics.py --net "$net"  --num_classes $NUMCLASSES --attack "$att" --detector "$det" --img_size 64 --eps "$eps"
                    fi 
                    if [ "$net" == celebaHQ128 ]; then
                        python -u extract_characteristics.py --net "$net"  --num_classes $NUMCLASSES --attack "$att" --detector "$det" --img_size 128 --eps "$eps" --wanted_samples 1600
                    fi 

                    if [ "$net" == celebaHQ256 ]; then
                        python -u extract_characteristics.py --net "$net"  --num_classes $NUMCLASSES --attack "$att" --detector "$det" --img_size 256 --eps "$eps" --wanted_samples 1500
                    fi 
                done
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
                            for eps in $EPSILONS; do
                                python -u detect_adversarials.py --net "$net" --num_classes $NUMCLASSES --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier"  --eps "$eps"
                            done
                        done
                    done
                done
            done 
    done
}

# genereratecleandata
attacks
# extractcharacteristics
# detectadversarials


# python attacks.py --net celebaHQ32 --att std --batch_size 500 --num_classes 2 --eps 4./255.
# python attacks.py --net celebaHQ32 --att std --batch_size 500 --num_classes 2 --eps 2./255.
# python attacks.py --net celebaHQ32 --att std --batch_size 500 --num_classes 2 --eps 1./255.
# python attacks.py --net celebaHQ32 --att std --batch_size 500 --num_classes 2 --eps 0.5/255.


# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack fgsm
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack bim
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack std
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack pgd
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack df
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack cw

# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack fgsm
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack bim
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack std
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack pgd
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack df
# python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack cw



# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack fgsm
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack bim
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack std
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack pgd
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack df
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack cw

# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack fgsm
# sleep 3
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack bim
# sleep 3
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack std
# sleep 3
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack pgd
# sleep 3
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack df
# sleep 3
# python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack cw


# for nr in {1,2,3,4}; do
#     echo "Generate Clean Data:  run: $nr" 
#     python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/clean_data', net=['celebaHQ32', 'celebaHQ64', 'celebaHQ128'] )"
#     genereratecleandata
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/clean_data', net=['celebaHQ32', 'celebaHQ64', 'celebaHQ128'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"
# done

# for nr in 4; do
#     echo "Attacks:  run: $nr" 
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data',       net=['celebaHQ32', 'celebaHQ64', 'celebaHQ128'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"
#     # attacks
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',           net=['celebaHQ32', 'celebaHQ64', 'celebaHQ128'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"

#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data',       net=['celebaHQ32', 'celebaHQ64'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"
#     # attacks
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',  net=['celebaHQ32', 'celebaHQ64'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"

#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data',       net=['celebaHQ64'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"
#     # attacks
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',  net=['celebaHQ64'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"
# done


# python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/clean_data', net=['imagenet32', 'imagenet64', 'imagenet128'] )"
# python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/attacks',    net=['imagenet32', 'imagenet64', 'imagenet128'] )"

# # genereratecleandata
# # attacks

# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/clean_data', net=['celebaHQ32', 'celebaHQ64', 'celebaHQ128'], dest='./log_evaluation/celebAHQ', run_nr=2)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',    net=['celebaHQ32', 'celebaHQ64', 'celebaHQ128'], dest='./log_evaluation/celebAHQ', run_nr=2)"


# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics', net=['celebaHQ32'], dest='./log_evaluation/celebAHQ', run_nr=1)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection', net=['celebaHQ32'], dest='./log_evaluation/celebAHQ', run_nr=1)"

# extractcharacteristics
# detectadversarials




# celebaHQ64
# for nr in 1; do
#     python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['celebaHQ64'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"
#     extractcharacteristics
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics', net=['celebaHQ64'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"
# done

# for nr in 1; do
#     detectadversarials
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection', net=['celebaHQ64'], dest='./log_evaluation/celebAHQ', run_nr=$nr)"
# done 



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



python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputPFS --attack fgsm
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputPFS --attack bim
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputPFS --attack std
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputPFS --attack pgd
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputPFS --attack df
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputPFS --attack cw


python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputMFS --attack fgsm
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputMFS --attack bim
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputMFS --attack std
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputMFS --attack pgd
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputMFS --attack df
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector InputMFS --attack cw


python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerPFS --attack fgsm
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerPFS --attack bim
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerPFS --attack std
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerPFS --attack pgd 
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerPFS --attack df
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerPFS --attack cw


python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerMFS --attack fgsm
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerMFS --attack bim
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerMFS --attack std
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerMFS --attack pgd
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerMFS --attack df
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LayerMFS --attack cw

python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LID --attack fgsm
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LID --attack bim
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LID --attack std
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LID --attack pgd
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LID --attack df
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector LID --attack cw

python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector Mahalanobis --attack fgsm
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector Mahalanobis --attack bim
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector Mahalanobis --attack std
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector Mahalanobis --attack pgd
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector Mahalanobis --attack df
python -u extract_characteristics.py --net celebaHQ32 --num_classes 2  --detector Mahalanobis --attack cw




python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack cw


python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack pgd 
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack cw

python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack fgsm
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack bim
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack std
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack pgd
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack df
python -u detect_adversarials.py --net celebaHQ32 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack cw





python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputPFS --attack fgsm
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputPFS --attack bim
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputPFS --attack std
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputPFS --attack pgd
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputPFS --attack df
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputPFS --attack cw


python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack fgsm
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack bim
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack std
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack pgd
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack df
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector InputMFS --attack cw


python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerPFS --attack fgsm
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerPFS --attack bim
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerPFS --attack std
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerPFS --attack pgd 
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerPFS --attack df
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerPFS --attack cw


python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack fgsm
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack bim
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack std
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack pgd
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack df
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LayerMFS --attack cw

python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LID --attack fgsm
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LID --attack bim
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LID --attack std
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LID --attack pgd
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LID --attack df
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector LID --attack cw

python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector Mahalanobis --attack fgsm
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector Mahalanobis --attack bim
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector Mahalanobis --attack std
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector Mahalanobis --attack pgd
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector Mahalanobis --attack df
python -u extract_characteristics.py --net celebaHQ64 --num_classes 2 --img_size 64 --detector Mahalanobis --attack cw




python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --attack cw


python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack pgd 
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID --attack cw

python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack fgsm
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack bim
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack std
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack pgd
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack df
python -u detect_adversarials.py --net celebaHQ64 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --attack cw















python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputPFS --img_size 128  --attack fgsm
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputPFS --img_size 128  --attack bim
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputPFS --img_size 128  --attack std
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputPFS --img_size 128  --attack pgd
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputPFS --img_size 128  --attack df
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputPFS --img_size 128  --attack cw


python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack fgsm
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack bim
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack std
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack pgd
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack df
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector InputMFS --img_size 128  --attack cw


python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerPFS --img_size 128  --attack fgsm
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerPFS --img_size 128  --attack bim
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerPFS --img_size 128  --attack std
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerPFS --img_size 128  --attack pgd 
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerPFS --img_size 128  --attack df
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerPFS --img_size 128  --attack cw

python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack fgsm
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack bim
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack std
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack pgd
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack df
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LayerMFS --img_size 128 --attack cw

python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LID --img_size 128  --attack fgsm
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LID --img_size 128  --attack bim
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LID --img_size 128  --attack std
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LID --img_size 128  --attack pgd
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LID --img_size 128  --attack df
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector LID --img_size 128  --attack cw

python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector Mahalanobis --img_size 128  --attack fgsm
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector Mahalanobis --img_size 128  --attack bim
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector Mahalanobis --img_size 128  --attack std
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector Mahalanobis --img_size 128  --attack pgd
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector Mahalanobis --img_size 128  --attack df
python -u extract_characteristics.py --net celebaHQ128 --num_classes 2  --detector Mahalanobis --img_size 128  --attack cw










python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --img_size 128 --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --img_size 128 --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --img_size 128 --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --img_size 128 --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --img_size 128 --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputPFS --img_size 128 --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --img_size 128 --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --img_size 128 --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --img_size 128 --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --img_size 128 --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --img_size 128 --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector InputMFS --img_size 128 --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --img_size 128 --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --img_size 128 --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --img_size 128 --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --img_size 128 --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --img_size 128 --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerPFS --img_size 128 --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --img_size 128 --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --img_size 128 --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --img_size 128 --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --img_size 128 --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --img_size 128 --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LayerMFS --img_size 128 --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --img_size 128  --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --img_size 128  --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --img_size 128  --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --img_size 128  --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --img_size 128  --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector LID --img_size 128  --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --img_size 128  --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --img_size 128  --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --img_size 128  --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --img_size 128  --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --img_size 128  --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf LR --detector Mahalanobis --img_size 128  --attack cw


python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --img_size 128  --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --img_size 128  --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --img_size 128  --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --img_size 128  --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --img_size 128  --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputPFS --img_size 128  --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --img_size 128  --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --img_size 128  --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --img_size 128  --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --img_size 128  --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --img_size 128  --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector InputMFS --img_size 128  --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --img_size 128  --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --img_size 128  --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --img_size 128  --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --img_size 128  --attack pgd 
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --img_size 128  --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerPFS --img_size 128  --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --img_size 128  --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --img_size 128  --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --img_size 128  --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --img_size 128  --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --img_size 128  --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LayerMFS --img_size 128  --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID  --img_size 128  --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID  --img_size 128  --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID  --img_size 128  --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID  --img_size 128  --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID  --img_size 128  --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector LID  --img_size 128  --attack cw

python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --img_size 128  --attack fgsm
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --img_size 128  --attack bim
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --img_size 128  --attack std
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --img_size 128  --attack pgd
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --img_size 128  --attack df
python -u detect_adversarials.py --net celebaHQ128 --num_classes 2  --wanted_samples 1500 --clf RF --detector Mahalanobis --img_size 128  --attack cw