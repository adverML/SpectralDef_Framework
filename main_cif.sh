#!/bin/bash

######## To Clarify from the Paper
# InputMFS == BlackBox_MFS
# InputPFS == BlackBox_PFS
# LayerMFS == WhiteBox_MFS
# LayerPFS == WhiteBox_PFS

function log_msg {
  echo  "`date` $@"
}

# DATASETS=(cif10 cif10vgg cif100 cif100vgg imagenet imagenet32 imagenet64 imagenet128 celebaHQ32 celebaHQ64 celebaHQ128)
# DATASETS="cif10 cif10vgg cif100 cif100vgg"
DATASETS="cif10"
RUNS="1 2 3"

# ATTACKS="fgsm bim pgd std df cw"
ATTACKS="fgsm bim pgd std df cw"
# DETECTORS="InputPFS LayerPFS LID Mahalanobis"
# DETECTORS="InputPFS LayerPFS InputMFS LayerMFS LID Mahalanobis"
DETECTORS="LayerMFS"
# EPSILONS="8./255. 4./255. 2./255. 1./255. 0.5/255."
EPSILONS="8./255."


CLF="LR RF"
# CLF="LR"

IMAGENET32CLASSES="25 50 100 250 1000"
# NRSAMPLES="300 500 1000 1200 1500 2000" # only at detectadversarialslayer
NRSAMPLES="1500"


DATASETSLAYERNR="cif10 cif10vgg"
ATTACKSLAYERNR=" df"
# ATTACKSLAYERNR="fgsm bim pgd std df cw"
NRRUN=5
LAYERNR={0,1,2,3,4,5,6,7,8,9,10,11,12}
DETECTORSLAYERNR="LayerMFS LayerPFS"

#-----------------------------------------------------------------------------------------------------------------------------------
log_msg "Networks are already trained!"
#-----------------------------------------------------------------------------------------------------------------------------------

printn()
{
    for index in $RUNS; do
        echo "$index"
    done 
}


genereratecleandata ()
{
    log_msg "Generate Clean Data for Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            if [ "$net" == cif10 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run"
            fi 

            if [ "$net" == cif10vgg ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 10  --run_nr "$run"
            fi 

            if [ "$net" == cif100 ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run"
            fi 

            if [ "$net" == cif100vgg ]; then
                python -u generate_clean_data.py --net "$net" --num_classes 100  --run_nr "$run"
            fi 
        done
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
attacks ()
{
    log_msg "Attack Clean Data with Foolbox Attacks and Autoattack!"
    for run in $RUNS; do
        for net in $DATASETS; do
            for att in $ATTACKS; do
                for eps in $EPSILONS; do
                    if [ "$net" == cif10 ]; then
                        if  [ "$att" == std ]; then                                
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run"
                        else
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --net_normalization --run_nr "$run"
                        fi
                    fi

                    if [ "$net" == cif10vgg ]; then
                        if  [ "$att" == std ]; then                                
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 500 --eps "$eps"  --run_nr "$run"
                        else
                            python -u attacks.py --net "$net" --num_classes 10 --attack "$att" --img_size 32 --batch_size 1500  --net_normalization --run_nr "$run"
                        fi
    
                    fi 

                    if [ "$net" == cif100 ]; then
                        python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --run_nr "$run"
                    fi 

                    if [ "$net" == cif100vgg ]; then
                        python -u attacks.py --net "$net" --num_classes 100 --attack "$att" --img_size 32 --batch_size 1000  --run_nr "$run"
                    fi 
                done
            done
        done
    done
}

#-----------------------------------------------------------------------------------------------------------------------------------
extractcharacteristics ()
{
    log_msg "Extract Characteristics"
    for run in $RUNS; do
        for net in $DATASETS; do
            for att in $ATTACKS; do
                for eps in $EPSILONS; do
                    for det in $DETECTORS; do
                            if [ "$net" == cif10 ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det"  --num_classes 10 --eps "$eps" --run_nr "$run" --take_inputimage 
                            fi

                            if [ "$net" == cif10vgg ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10   --run_nr "$run" --take_inputimage 
                            fi 

                            if [ "$net" == cif100 ]; then
                                if  [ "$att" == std ]; then                                
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --eps "$eps"  --run_nr "$run" 
                                else 
                                    python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100  --run_nr "$run"
                                fi
                            fi

                            if [ "$net" == cif100vgg ]; then
                                python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 100   --run_nr "$run" 
                            fi 
                    done
                done
            done
        done
    done
}


extractcharacteristicslayer ()
{
    log_msg "Extract Characteristics Layer By Layer for WhiteBox"
    for run in $RUNS; do
        for net in $DATASETSLAYERNR; do
            for att in $ATTACKSLAYERNR; do
                for det in $DETECTORSLAYERNR; do
                    for nr in {0,1,2,3,4,5,6,7,8,9,10,11,12}; do 
                        log_msg "Layer Nr. $nr; attack $att; detectors $det"
                        if [ "$net" == cif10 ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr "$run"   --take_inputimage 
                        fi

                        if [ "$net" == cif10vgg ]; then
                            python -u extract_characteristics.py --net "$net" --attack "$att" --detector "$det" --num_classes 10 --nr "$nr" --run_nr "$run"   --take_inputimage 
                        fi 
                    done
                done
            done
        done
    done
}

# #-----------------------------------------------------------------------------------------------------------------------------------
detectadversarials ()
{
    log_msg "Detect Adversarials!"
    for run in $RUNS; do
        for net in $DATASETS; do
                for att in $ATTACKS; do
                    for eps in $EPSILONS; do
                        for det in $DETECTORS; do
                            for nrsamples in $NRSAMPLES; do
                                for classifier in $CLF; do
                                        if [ "$net" == cif10 ]; then
                                            python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --eps "$eps" --num_classes 10  --run_nr "$run"
                                        fi

                                        if [ "$net" == cif10vgg ]; then
                                            python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10  --run_nr "$run"
                                        fi 

                                        if [ "$net" == cif100 ]; then
                                            if  [ "$att" == std ]; then
                                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100 --eps "$eps"  --run_nr "$run"
                                            else
                                                python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100  --run_nr "$run"
                                            fi
                                        fi

                                        if [ "$net" == cif100vgg ]; then
                                            python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 100   --run_nr "$run"
                                        fi 
                                done
                            done
                        done
                    done
                done
        done
    done
}


detectadversarialslayer ()
{
    log_msg "Detect Adversarials Layer By Layer!"
    for net in $DATASETSLAYERNR; do
            for att in $ATTACKSLAYERNR; do
                for det in $DETECTORSLAYERNR; do
                    for nrsamples in $NRSAMPLES; do
                        for classifier in $CLF; do
                            for nr in {0,1,2,3,4,5,6,7,8,9,10,11,12}; do 
                                log_msg "Layer Nr. $nr"
                                if [ "$net" == cif10 ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr  "$NRRUN" 
                                fi

                                if [ "$net" == cif10vgg ]; then
                                    python -u detect_adversarials.py --net "$net" --attack "$att" --detector "$det" --wanted_samples "$nrsamples" --clf "$classifier" --num_classes 10 --nr "$nr" --run_nr  "$NRRUN" 
                                fi 
                            done
                        done
                    done
                done
            done
    done
}


# extractcharacteristicslayer
# detectadversarialslayer

# printn
# genereratecleandata
attacks
# extractcharacteristics
# detectadversarials


# python attacks.py --net cif10 --att std --batch_size 500 --eps 4./255.
# python attacks.py --net cif10 --att std --batch_size 500 --eps 2./255.
# python attacks.py --net cif10 --att std --batch_size 500 --eps 1./255.
# python attacks.py --net cif10 --att std --batch_size 500 --eps 0.5/255.


# python attacks.py --net cif10vgg --att std --batch_size 500 --eps 4./255.
# python attacks.py --net cif10vgg --att std --batch_size 500 --eps 2./255.
# python attacks.py --net cif10vgg --att std --batch_size 500 --eps 1./255.
# python attacks.py --net cif10vgg --att std --batch_size 500 --eps 0.5/255.


# for nr in $NRRUN; do

#     python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/clean_data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'] )"
#     python -c "import evaluate_detection; evaluate_detection.clean_root_folders( root='./data/attacks',    net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'] )"

#     genereratecleandata

#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/clean_data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"

#     attacks

#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/attacks',    net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
# done




# Layer Nr
# python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['cif10', 'cif10vgg'], dest='./log_evaluation/cif', run_nr=1)"
# extractcharacteristicslayer
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics',    net=['cif10', 'cif10vgg'], dest='./log_evaluation/cif', run_nr=1)"


# detectadversarialslayer
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection',    net=['cif10vgg'], dest='./log_evaluation/cif', run_nr=1)"

# for nr in {1,2,3,4}; do
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data/', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
#     # detectadversarialslayer
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection',    net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
#     python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data/',          net=['cif10'], dest='./log_evaluation/cif', run_nr=$nr)"
#     detectadversarialslayer
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection',    net=['cif10'], dest='./log_evaluation/cif', run_nr=$nr)"
# done


# for nr in {2,3}; do
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
#     # extractcharacteristics
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics',    net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"

#     python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['cif10'], dest='./log_evaluation/cif', run_nr=$nr)"
#     extractcharacteristics
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics', net=['cif10'], dest='./log_evaluation/cif', run_nr=$nr)"
# done


# for nr in $NRRUN; do
#     python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data/', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
#     detectadversarials
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection',    net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
# done


# for nr in 1; do
#     # python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data/', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
#     detectadversarials
#     python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection',    net=['cif10vgg'], dest='./log_evaluation/cif', run_nr=$nr)"
# done


# #-----------------------------------------------------------------------------------------------------------------------------------
# python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data/', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=1)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data', net=['cif10'], dest='./log_evaluation/cif', run_nr=1)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection',    net=['cif10'], dest='./log_evaluation/cif', run_nr=1)"

# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/extracted_characteristics', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=1)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection',                 net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=1)"


# python -u extract_characteristics.py --net cif100vgg  --detector InputPFS   --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100vgg  --detector InputPFS   --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100vgg  --detector InputPFS   --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100vgg  --detector InputPFS   --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100vgg  --detector InputPFS   --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100vgg  --detector InputPFS   --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100vgg  --detector InputMFS   --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100vgg  --detector InputMFS   --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100vgg  --detector InputMFS   --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100vgg  --detector InputMFS   --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100vgg  --detector InputMFS   --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100vgg  --detector InputMFS   --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100vgg  --detector LayerPFS   --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100vgg  --detector LayerPFS   --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100vgg  --detector LayerPFS   --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100vgg  --detector LayerPFS   --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100vgg  --detector LayerPFS   --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100vgg  --detector LayerPFS   --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100vgg  --detector LayerMFS   --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100vgg  --detector LayerMFS   --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100vgg  --detector LayerMFS   --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100vgg  --detector LayerMFS   --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100vgg  --detector LayerMFS   --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100vgg  --detector LayerMFS   --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100vgg  --detector LID        --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100vgg  --detector LID        --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100vgg  --detector LID        --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100vgg  --detector LID        --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100vgg  --detector LID        --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100vgg  --detector LID        --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100vgg  --detector Mahalanobis --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100vgg  --detector Mahalanobis --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100vgg  --detector Mahalanobis --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100vgg  --detector Mahalanobis --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100vgg  --detector Mahalanobis --num_classes 100 --attack df o
# python -u extract_characteristics.py --net cif100vgg  --detector Mahalanobis --num_classes 100 --attack cw o

# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100 

# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100  

# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100  

# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100

# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector LID       --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector LID       --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector LID       --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector LID       --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector LID       --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector LID       --wanted_samples 1500 --clf LR --num_classes 100  

# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100  



# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100
# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector LID       --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector LID       --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector LID       --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector LID       --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector LID       --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector LID       --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack fgsm --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack bim  --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100vgg --attack std  --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack pgd  --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack df   --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100vgg --attack cw   --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100











# python -u extract_characteristics.py --net cif100  --detector InputPFS   --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100  --detector InputPFS   --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100  --detector InputPFS   --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100  --detector InputPFS   --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100  --detector InputPFS   --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100  --detector InputPFS   --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100  --detector InputMFS   --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100  --detector InputMFS   --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100  --detector InputMFS   --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100  --detector InputMFS   --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100  --detector InputMFS   --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100  --detector InputMFS   --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100  --detector LayerPFS   --num_classes 100 --attack fgsm 
# python -u extract_characteristics.py --net cif100  --detector LayerPFS   --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100  --detector LayerPFS   --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100  --detector LayerPFS   --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100  --detector LayerPFS   --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100  --detector LayerPFS   --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100  --detector LayerMFS   --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100  --detector LayerMFS   --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100  --detector LayerMFS   --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100  --detector LayerMFS   --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100  --detector LayerMFS   --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100  --detector LayerMFS   --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100  --detector LID        --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100  --detector LID        --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100  --detector LID        --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100  --detector LID        --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100  --detector LID        --num_classes 100 --attack df
# python -u extract_characteristics.py --net cif100  --detector LID        --num_classes 100 --attack cw  

# python -u extract_characteristics.py --net cif100  --detector Mahalanobis --num_classes 100 --attack fgsm
# python -u extract_characteristics.py --net cif100  --detector Mahalanobis --num_classes 100 --attack bim 
# python -u extract_characteristics.py --net cif100  --detector Mahalanobis --num_classes 100 --attack std 
# python -u extract_characteristics.py --net cif100  --detector Mahalanobis --num_classes 100 --attack pgd 
# python -u extract_characteristics.py --net cif100  --detector Mahalanobis --num_classes 100 --attack df 
# python -u extract_characteristics.py --net cif100  --detector Mahalanobis --num_classes 100 --attack cw 

# python -u detect_adversarials.py --net cif100 --attack fgsm --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 100 

# python -u detect_adversarials.py --net cif100 --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 100  

# python -u detect_adversarials.py --net cif100 --attack fgsm --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 100  

# python -u detect_adversarials.py --net cif100 --attack fgsm --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100
# python -u detect_adversarials.py --net cif100 --attack bim  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100
# python -u detect_adversarials.py --net cif100 --attack std  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack df   --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack cw   --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 100

# python -u detect_adversarials.py --net cif100 --attack fgsm --detector LID       --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector LID       --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector LID       --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector LID       --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector LID       --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector LID       --wanted_samples 1500 --clf LR --num_classes 100  

# python -u detect_adversarials.py --net cif100 --attack fgsm --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 100 



# python -u detect_adversarials.py --net cif100 --attack fgsm --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector InputPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack fgsm --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector InputMFS  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack fgsm --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100
# python -u detect_adversarials.py --net cif100 --attack bim  --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100
# python -u detect_adversarials.py --net cif100 --attack std  --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack df   --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack cw   --detector LayerMFS  --wanted_samples 1500 --clf RF --num_classes 100
# python -u detect_adversarials.py --net cif100 --attack fgsm --detector LID       --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector LID       --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector LID       --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector LID       --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector LID       --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector LID       --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack fgsm --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack bim  --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100 
# python -u detect_adversarials.py --net cif100 --attack std  --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack pgd  --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack df   --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100  
# python -u detect_adversarials.py --net cif100 --attack cw   --detector Mahalanobis  --wanted_samples 1500 --clf RF --num_classes 100





# python -u extract_characteristics.py --net cif10vgg  --detector InputPFS   --num_classes 10 --attack fgsm
# python -u extract_characteristics.py --net cif10vgg  --detector InputPFS   --num_classes 10 --attack bim 
# python -u extract_characteristics.py --net cif10vgg  --detector InputPFS   --num_classes 10 --attack std 
# python -u extract_characteristics.py --net cif10vgg  --detector InputPFS   --num_classes 10 --attack pgd 
# python -u extract_characteristics.py --net cif10vgg  --detector InputPFS   --num_classes 10 --attack df
# python -u extract_characteristics.py --net cif10vgg  --detector InputPFS   --num_classes 10 --attack cw  

# python -u extract_characteristics.py --net cif10vgg  --detector InputMFS   --num_classes 10 --attack fgsm
# python -u extract_characteristics.py --net cif10vgg  --detector InputMFS   --num_classes 10 --attack bim 
# python -u extract_characteristics.py --net cif10vgg  --detector InputMFS   --num_classes 10 --attack std 
# python -u extract_characteristics.py --net cif10vgg  --detector InputMFS   --num_classes 10 --attack pgd 
# python -u extract_characteristics.py --net cif10vgg  --detector InputMFS   --num_classes 10 --attack df
# python -u extract_characteristics.py --net cif10vgg  --detector InputMFS   --num_classes 10 --attack cw  

# python -u extract_characteristics.py --net cif10vgg  --detector LayerPFS   --num_classes 10 --attack fgsm
# python -u extract_characteristics.py --net cif10vgg  --detector LayerPFS   --num_classes 10 --attack bim 
# python -u extract_characteristics.py --net cif10vgg  --detector LayerPFS   --num_classes 10 --attack std 
# python -u extract_characteristics.py --net cif10vgg  --detector LayerPFS   --num_classes 10 --attack pgd 
# python -u extract_characteristics.py --net cif10vgg  --detector LayerPFS   --num_classes 10 --attack df
# python -u extract_characteristics.py --net cif10vgg  --detector LayerPFS   --num_classes 10 --attack cw  

# python -u extract_characteristics.py --net cif10vgg  --detector LayerMFS   --num_classes 10 --attack fgsm
# python -u extract_characteristics.py --net cif10vgg  --detector LayerMFS   --num_classes 10 --attack bim 
# python -u extract_characteristics.py --net cif10vgg  --detector LayerMFS   --num_classes 10 --attack std 
# python -u extract_characteristics.py --net cif10vgg  --detector LayerMFS   --num_classes 10 --attack pgd 
# python -u extract_characteristics.py --net cif10vgg  --detector LayerMFS   --num_classes 10 --attack df
# python -u extract_characteristics.py --net cif10vgg  --detector LayerMFS   --num_classes 10 --attack cw  

# python -u extract_characteristics.py --net cif10vgg  --detector LID        --num_classes 10 --attack fgsm
# python -u extract_characteristics.py --net cif10vgg  --detector LID        --num_classes 10 --attack bim 
# python -u extract_characteristics.py --net cif10vgg  --detector LID        --num_classes 10 --attack std 
# python -u extract_characteristics.py --net cif10vgg  --detector LID        --num_classes 10 --attack pgd 
# python -u extract_characteristics.py --net cif10vgg  --detector LID        --num_classes 10 --attack df
# python -u extract_characteristics.py --net cif10vgg  --detector LID        --num_classes 10 --attack cw  

# python -u extract_characteristics.py --net cif10vgg  --detector Mahalanobis --num_classes 10 --attack fgsm
# python -u extract_characteristics.py --net cif10vgg  --detector Mahalanobis --num_classes 10 --attack bim 
# python -u extract_characteristics.py --net cif10vgg  --detector Mahalanobis --num_classes 10 --attack std 
# python -u extract_characteristics.py --net cif10vgg  --detector Mahalanobis --num_classes 10 --attack pgd 
# python -u extract_characteristics.py --net cif10vgg  --detector Mahalanobis --num_classes 10 --attack df
# python -u extract_characteristics.py --net cif10vgg  --detector Mahalanobis --num_classes 10 --attack cw  



# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack bim  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack std  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack pgd  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack df   --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack cw   --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10 

# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack bim  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack std  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack pgd  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack df   --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack cw   --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10  

# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack bim  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack std  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack pgd  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack df   --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack cw   --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10  

# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack bim  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack std  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack pgd  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack df   --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack cw   --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10 

# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LID       --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack bim  --detector LID       --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack std  --detector LID       --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack pgd  --detector LID       --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack df   --detector LID       --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack cw   --detector LID       --wanted_samples 1500 --clf LR --num_classes 10  

# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack bim  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10vgg --attack std  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack pgd  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack df   --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10vgg --attack cw   --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10  




# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 6  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 7  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack fgsm  --detector LayerPFS --net cif10vgg --nr 12


# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 6   
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 7  
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 9   
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack bim  --detector LayerPFS --net cif10vgg --nr 12


# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 6  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 7  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack std  --detector LayerPFS --net cif10vgg --nr 12


# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 6  
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 7   
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack pgd  --detector LayerPFS --net cif10vgg --nr 12



# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 6  
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 7   
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 10 
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack df  --detector LayerPFS --net cif10vgg --nr 12


# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 6  
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 7   
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack cw  --detector LayerPFS --net cif10vgg --nr 12






# ßßßßßßßßßßßßßßßßßßßßßßßßßßßß

# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 6  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 7  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack fgsm  --detector LayerMFS --net cif10vgg --nr 12

# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 6   
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 7  
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 9   
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack bim  --detector LayerMFS --net cif10vgg --nr 12

# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 6  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 7  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack std  --detector LayerMFS --net cif10vgg --nr 12

# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 1 
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 6 
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 7 
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack pgd  --detector LayerMFS --net cif10vgg --nr 12

# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 6  
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 7   
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 10 
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack df  --detector LayerMFS --net cif10vgg --nr 12

# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 0  
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 1  
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 2  
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 3  
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 4  
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 5  
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 6  
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 7   
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 8  
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 9  
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 10
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 11
# python extract_characteristics.py  --attack cw  --detector LayerMFS --net cif10vgg --nr 12

# python -u detect_adversarials.py --net cif10 --attack fgsm --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack bim  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack std  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack pgd  --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack df   --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack cw   --detector InputPFS  --wanted_samples 1500 --clf LR --num_classes 10 

# python -u detect_adversarials.py --net cif10 --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack bim  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack std  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack pgd  --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack df   --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack cw   --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10  

# python -u detect_adversarials.py --net cif10 --attack fgsm --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack bim  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack std  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack pgd  --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack df   --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack cw   --detector InputMFS  --wanted_samples 1500 --clf LR --num_classes 10  

# python -u detect_adversarials.py --net cif10 --attack fgsm --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack bim  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack std  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack pgd  --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack df   --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack cw   --detector LayerMFS  --wanted_samples 1500 --clf LR --num_classes 10 

# python -u detect_adversarials.py --net cif10 --attack fgsm --detector LID       --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack bim  --detector LID       --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack std  --detector LID       --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack pgd  --detector LID       --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack df   --detector LID       --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack cw   --detector LID       --wanted_samples 1500 --clf LR --num_classes 10  

# python -u detect_adversarials.py --net cif10 --attack fgsm --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack bim  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10 
# python -u detect_adversarials.py --net cif10 --attack std  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack pgd  --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack df   --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10  
# python -u detect_adversarials.py --net cif10 --attack cw   --detector Mahalanobis  --wanted_samples 1500 --clf LR --num_classes 10  


# #########################################################
# #########################################################


# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 0
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 4
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 6 
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 7
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 9
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 10
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 12 


# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 0
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 4 
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 6 
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 7
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 9
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 10
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 11 
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 12 



# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 0
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 4
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 6 
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 7
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 9
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 10
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 12


 
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 0 
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 4
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 6  
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 7  

# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 9
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 10
    # python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 12



# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 0 
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 4
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 6   
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 7
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 9
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 10
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 12


# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 0 
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 4 
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 6 
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 7
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 9
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 10
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf LR --num_classes 10 --nr 12



# #########################################################
# #########################################################


# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 0
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 4
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 6 
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 7 
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 9
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 10
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack fgsm --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 12 


# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 0 
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 1 
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 4 
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 6 
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 7 
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 9  
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 10
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 11 
# python -u detect_adversarials.py --net cif10vgg --attack bim --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 12 



# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 0 
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 2 
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 4
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 6  
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 7
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 9
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 10
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack std --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 12


 
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 0  
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 4
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 6  
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 7  
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 8 
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 9  
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 10
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack pgd --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 12 



# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 0 
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 2
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 4
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 6  
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 7
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 9   
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 10  
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack df --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 12


# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 0     
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 1
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 2   
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 3
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 4 
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 5
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 6 
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 7
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 8
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 9
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 10 
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 11
# python -u detect_adversarials.py --net cif10vgg --attack cw --detector LayerPFS  --wanted_samples 1500 --clf RF --num_classes 10 --nr 12 


# python -u extract_characteristics.py --net cif10 --attack fgsm  --detector Mahannobis --num_classes 10



# #-----------------------------------------------------------------------------------------------------------------------------------
log_msg "finished"
exit 0
