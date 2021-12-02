#!/bin/bash


# bash main_imagnet128.sh  &> log_evaluation/imagenet128/all.log

function log_msg {
  echo "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
DATASETS="imagenet128"
# ATTACKS="fgsm bim pgd std df cw"
ATTACKS="cw"
# ATTACKS="fgsm bim pgd df cw"
# ATTACKS="fgsm bim pgd std df"

EPSILONS="4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."


# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
DETECTORS="LayerMFS"
CLF="LR"
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

        if [ "$net" == imagenet128 ]; then
            python -u generate_clean_data.py --net "$net" --num_classes 1000   --img_size 128
        fi 
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
attacks ()
{
    log_msg "Attack Clean Data with Foolbox Attacks and Autoattack!"

    for net in $DATASETS; do

        for att in $ATTACKS; do

             if [ "$net" == imagenet128 ]; then
                python -u attacks.py --net "$net" --num_classes 1000 --attack "$att" --img_size 128 --batch_size 64
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
            for eps in $EPSILONS; do
                for det in $DETECTORS; do
                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 1000 --img_size 128 --wanted_samples 1600 --eps "$eps"
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
                for eps in $EPSILONS; do
                    for det in $DETECTORS; do
                        for nrsamples in $NRSAMPLES; do
                            for classifier in $CLF; do
                                python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 1000 --eps "$eps"
                            done
                        done
                    done
                done
            done
    done
}

# extractcharacteristics
detectadversarials


# python attacks.py --net imagenet128 --att std --batch_size 64 --num_classes 1000 --img_size 128 --eps 4./255.
# python attacks.py --net imagenet128 --att std --batch_size 64 --num_classes 1000 --img_size 128 --eps 2./255.
# python attacks.py --net imagenet128 --att std --batch_size 64 --num_classes 1000 --img_size 128 --eps 1./255.
# python attacks.py --net imagenet128 --att std --batch_size 64 --num_classes 1000 --img_size 128 --eps 0.5/255.



# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack fgsm
# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack bim
# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack std
# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack pgd
# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack df
# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack cw


# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack fgsm
# sleep 3

# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack bim
# sleep 3

# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack std
# sleep 3

# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack pgd
# sleep 3

# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack df
# sleep 3

# python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack cw



# for nr in {1,2,3,4}; do
#     echo "Generate Clean Data:  run: $nr" 

#     python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/clean_data', net=['imagenet128'] )"
#     genereratecleandata
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/clean_data', net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=$nr)"
    
# done


# for nr in 4; do
#     echo "Attacks:  run: $nr" 
#     python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data',       net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=$nr)"
#     attacks
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',  net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=$nr)"
# done



# imagenet128
# for nr in 1; do
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=$nr)"
#     extractcharacteristics
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics', net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=$nr)"
# done

# for nr in 1; do
#     detectadversarials
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection', net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=$nr)"
# done 

# python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/clean_data', net=['imagenet128'] )"
# python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/attacks',    net=['imagenet128'] )"

# genereratecleandata
# attacks

# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/clean_data', net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=2)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',    net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=2)"

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
# [] Copy / Save in file structure
# [] Create CSV

# [] Variance
#   [] Run 1
#   [x] Run 2 
#       [x] copied
#   [] Run 3




python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack fgsm
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack bim
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack std
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack pgd
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack df
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector InputMFS --img_size 128  --attack cw


python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack fgsm
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack bim
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack std
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack pgd
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack df
python -u extract_characteristics.py --net imagenet128 --num_classes 1000  --detector LayerMFS --img_size 128 --attack cw