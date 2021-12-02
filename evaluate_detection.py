#!/usr/bin/env python3
# """ Evaluate Detection

# author Peter Lorenz
# """

import os
import glob
import pdb 
import shutil
import tqdm
import pandas as pd
import numpy as np
from pathlib import Path


from conf import settings
# import csv
# import argparse


def check_eps(str_val):
    if '8./255.' in str_val:
        eps = '8_255'
    elif '4./255.' in str_val:
        eps = '4_255'     
    elif '2./255.' in str_val:
        eps = '2_255'
    elif '1./255.' in str_val:
        eps = '1_255'
    elif '0.5/255.' in str_val:
        eps = '05_255'

    return eps


def is_float(value):
  try:
    float(value)
    return True
  except:
    return False


def get_clean_accuracy(paths):
    result = {}
    for path in paths:
        if 'fgsm' in path:
            attack_method = 'fgsm'
        elif 'bim' in path:
            attack_method = 'bim'
        elif 'pgd' in path:
            attack_method = 'pgd'
        elif 'std' in path:
            attack_method = 'std'
            if '8_255' in path:
                eps = '8_255'
            elif '4_255' in path:
                eps = '4_255'     
            elif '2_255' in path:
                eps = '2_255'
            elif '1_255' in path:
                eps = '1_255'
            elif '05_255' in path:
                eps = '05_255'
        elif 'df' in path:
            attack_method = 'df'
        elif 'cw' in path:
            attack_method = 'cw'
        else:
            raise NotImplementedError("Attack Method not implemented! {}".format(path))


        
        # pdb.set_trace()

        with open(path) as f_attack:
            lines_attack = f_attack.readlines()

        asr_list = []
        for line in lines_attack:
            if line.__contains__("INFO: attack success rate:"):
                print(path, line)
                asr = float(line.strip().split(' ')[-1])
                asr_list.append(asr)            

                if attack_method  == 'std':
                    result[attack_method + '_' + eps] = asr_list[-1]
                else:
                    result[attack_method] = asr_list[-1]
    
    return result


def sort_paths_by_layer(paths):
    # './log_evaluation/cif/cif10/run_1/data/detection/cif10/wrn_28_10_10/fgsm/LayerPFS/layer_0/LR/log.txt'

    sorted_paths = sorted(paths, key=lambda x: int(x.split('/')[-3].split('_')[-1]))

    return sorted_paths



def extract_information(root='./data', net=['cif10'], dest='./data/detection', run_nr=[1], csv_filename='eval.csv', layers=True):
    print( ' Extract information! ' )

    ATTACKS   = 'attacks'
    DETECTION = 'detection'    

    if len(run_nr) == 0:
        run_nr = ['']

    final = []  
    for nr in run_nr:
        if run_nr == '':
            print("Err: Run_nr: ", nr)
            return

        for net_path in net:
            in_dir_attacks = os.path.join( root, ATTACKS,   'run_' + str(nr), net_path )
            in_dir_detects = os.path.join( root, DETECTION, 'run_' + str(nr), net_path )

            # import pdb; pdb.set_trace()

            clean_acc = get_clean_accuracy( glob.glob( in_dir_attacks + "/**/log.txt", recursive=True ) )

            if layers:
                detectors = settings.DETECTOR_LIST_LAYERS
            else:
                detectors = settings.DETECTOR_LIST

            

            paths = []  
            for det in detectors:
                for classifier in settings.CLF:
                    for att in  settings.ATTACKS_LIST:
                            lr_paths = []
                            if layers:
                                if att == 'std':
                                    # import pdb; pdb.set_trace()
                                    search_path = in_dir_detects + "/**/" + att + "/**/" + det + "/layer_*/" + classifier + "/log.txt"
                                else:
                                    search_path = in_dir_detects + "/**/" + att + "/" + det + "/layer_*/" + classifier + "/log.txt"

                                lr_paths =  sort_paths_by_layer( glob.glob( search_path, recursive=True) ) 
                            else:
                                if att == 'std':
                                    search_path = in_dir_detects + "/**/" + att + "/**/" + det + "/" + classifier + "/log.txt"
                                    # import pdb; pdb.set_trace()
                                else:
                                    search_path = in_dir_detects + "/**/" + att + "/" + det + "/" + classifier + "/log.txt"

                                lr_paths = glob.glob( search_path, recursive=True)
                            print("lr_paths: ", lr_paths)
                            # import pdb; pdb.set_trace()

                            paths = paths + lr_paths 


            # print("paths: ")
            # import pdb; pdb.set_trace()

            index_selected = []
            asr_name = []
            train_error = []
            for path in paths:
                index = []
                line_split = []
                
                print("path: ", path)
                with open(path) as f:
                    lines = f.readlines()

                for line in lines:
                    if line.__contains__("RES"):
                        splitted_line = line.strip().split(',')[1:]                
                        # print("splitted_line: ", splitted_line)
                        if is_float( splitted_line[0] ):
                            line_split.append( [ float(item) for item in splitted_line ] )


                    # if line.__contains__("ZeroDivisionError"):
                    #     line_split.append( [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] )

                    if line.__contains__("OUTPUT_PATH_DIR:"):
                        index_split_list = line.strip().split(' ')
                        index.append( path )

                    if line.__contains__("train error:"):
                        train_error.append(line.strip().split(' ')[-1])

                    if line.__contains__("'attack':"):
                        for att in settings.ATTACKS_LIST:
                            if line.find(att)!=-1:
                                if att == 'std':
                                    tmp_eps = check_eps(line)
                                    att = att + '_' + tmp_eps
                                asr_name.append( att )                        

                print("line_split: ", line_split)
                if len(line_split) == 0:
                    line_split.append( [ -1, -1, -1, -1, -1, -1  ] )
                    index.append( index_split_list[1] )
                csv_line = line_split[-1]
                if not csv_line[-1] == -1:
                    fnr = float(csv_line[-1]) / 100.
                    asr = np.round(clean_acc[asr_name[-1]]*100, 2)
                    csv_line.append(asr)
                    asrd = np.round((fnr*asr), 2)
                    
                    if float(train_error[-1]) == 0.5:
                        asrd = str(asrd) + '*'
                    
                    csv_line.append(asrd)
                else:
                    csv_line.append(-1)
                    csv_line.append(-1)

                index_selected.append(index[-1])
                final.append(csv_line)

            output_path = os.path.join(  root, DETECTION, 'run_' + str(nr), net_path, csv_filename)
            print("output_path: ", output_path)

            # import pdb; pdb.set_trace()
            df = pd.DataFrame(final, columns=['auc','acc','pre','tpr', 'f1', 'fnr', 'asr', 'asrd'], index=index_selected)
            df.to_csv(output_path, sep=',')


def copy_run_to_root(root='./data/', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=2):
    print("INFO: Copy-to-Root: ", root, net, dest, run_nr)
    
    for net_path in net:
        in_dir  = os.path.join( dest, net_path, 'run_' + str(run_nr), 'data' )
        out_dir = os.path.join( root )

        print('in_dir:  ', in_dir )
        print('out_dir: ', out_dir)

        shutil.copytree(in_dir, out_dir, dirs_exist_ok=True)


def clean_root_folders(root='./data/clean_data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg']):
    print("INFO: delete: ", root, net)

    # import pdb; pdb.set_trace()

    print("------------------ net possibilities --------------------")
    print('cif10', 'cif10vgg', 'cif100', 'cif100vgg')
    print('imagenet')
    print('imagenet32', 'imagenet64', 'imagenet128')
    print('celebaHQ32', 'celebaHQ64', 'celebaHQ128')

    for net_path in net:
        pth = root + os.sep + net_path
        shutil.rmtree(pth)


def copy_run_dest(root='./data/clean_data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=2):
    print("INFO: ", root, net, dest, run_nr)

    # import pdb; pdb.set_trace()

    for net_path in net:
        out_dir = os.path.join( dest, net_path, 'run_' + str(run_nr) )
        parsed_paths = []
        pth = root + os.sep + net_path + "/**/log.txt"
        files_files_paths = glob.glob(pth, recursive = True)
        
        for text_file in files_files_paths:
            tmp_path =  os.path.split(text_file)[0]
            parsed_paths.append(tmp_path)
            print("parsed paths: ", parsed_paths)

            shutil.copytree(tmp_path, out_dir + tmp_path[1:], dirs_exist_ok=True)

# clean_root_folders( root='./data/clean_data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'] )
# clean_root_folders( root='./data/clean_data', net=['imagenet32', 'imagenet64', 'imagenet128'] )
# clean_root_folders( root='./data/attacks',    net=['imagenet32', 'imagenet64', 'imagenet128'] )




# for run in [1, 2, 3]:
#     copy_run_dest(root='./data/clean_data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=run)
#     copy_run_dest(root='./data/attacks',    net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=run)


# copy_run_dest(root='./data/clean_data', net=['imagenet32', 'imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=2)
# copy_run_dest(root='./data/attacks', net=['imagenet32', 'imagenet64'], dest='./log_evaluation/imagenet3264', run_nr=2)

# copy_run_dest(root='./data/clean_data', net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=2)
# copy_run_dest(root='./data/attacks', net=['imagenet128'], dest='./log_evaluation/imagenet128', run_nr=2)

# copy_run_dest(root='./data/clean_data', net=['imagenet'], dest='./log_evaluation/imagenet', run_nr=2)
# copy_run_dest(root='./data/attacks', net=['imagenet'], dest='./log_evaluation/imagenet', run_nr=2)

# copy_run_dest(root='./data/clean_data', net=['celebaHQ32', 'celebaHQ64', 'celebaHQ128'], dest='./log_evaluation/celebAHQ', run_nr=2)
# copy_run_dest(root='./data/attacks',    net=['celebaHQ32', 'celebaHQ64', 'celebaHQ128'], dest='./log_evaluation/celebAHQ', run_nr=2)


# copy_run_to_root(root='./data/clean_data',    net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=2)
# extract_information(root='./data/clean_data', net=['cif10', 'cif10vgg', 'cif100', 'cif100vgg'], dest='./log_evaluation/cif', run_nr=[2])

# python -c "import evaluate_detection; evaluate_detection.copy_run_to_root(root='./data/detection', net=['cif10vgg'], dest='./log_evaluation/cif', run_nr=1)"
# python -c "import evaluate_detection; evaluate_detection.copy_run_dest(root='./data/detection', net=['cif10vgg'], dest='./log_evaluation/cif', run_nr=1)"





# extract_information(root='./data/clean_data', net=['cif10'], dest='./log_evaluation/cif', run_nr=[1], csv_filename='layers.csv', whitebox=True)

# python -c "import evaluate_detection; evaluate_detection.extract_information(root='./data/clean_data', net=['cif10vgg'], dest='./log_evaluation/cif', run_nr=[1], csv_filename='layers.csv', layers=False)"



# python -c "import evaluate_detection; evaluate_detection.extract_information(root='./data/clean_data', net=['cif10vgg'], dest='./log_evaluation/cif', run_nr=[1], csv_filename='layers_cif10vgg.csv', whitebox=True)"


extract_information(root='./data', net=['cif10vgg'], dest='./data/detection', run_nr=[0], csv_filename='eval_cif10vgg.csv', layers=True)
# extract_information(root='./data', net=['cif10'], dest='./data/detection', run_nr=[0], csv_filename='eval_cif10.csv', layers=True)
# extract_information(root='./data', net=['cif100'], dest='./data/detection', run_nr=[1], csv_filename='eval_cif100.csv', layers=False)

# extract_information(root='./data', net=['cif10vgg'], dest='./data/detection', run_nr=[1], csv_filename='eval_cif10vgg.csv', layers=False)
# extract_information(root='./data', net=['cif100vgg'], dest='./data/detection', run_nr=[1], csv_filename='eval_cif100vgg.csv', layers=False)



# extract_information(root='./data', net=['imagenet32'], dest='./data/detection', run_nr=[1], csv_filename='eval_imagenet32.csv', layers=False)
# extract_information(root='./data', net=['imagenet64'], dest='./data/detection', run_nr=[1], csv_filename='eval_imagenet64.csv', layers=False)
# extract_information(root='./data', net=['imagenet128'], dest='./data/detection', run_nr=[1], csv_filename='eval_imagenet128.csv', layers=False)

# extract_information(root='./data', net=['celebaHQ32'], dest='./data/detection', run_nr=[1], csv_filename='eval_celebaHQ32.csv', layers=False)
# extract_information(root='./data', net=['celebaHQ64'], dest='./data/detection', run_nr=[1], csv_filename='eval_celebaHQ64.csv', layers=False)
# extract_information(root='./data', net=['celebaHQ128'], dest='./data/detection', run_nr=[1], csv_filename='eval_celebaHQ128.csv', layers=False)










