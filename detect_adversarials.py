#!/usr/bin/env python3
""" Detect Adversarials

author Peter Lorenz
"""
print('Load modules...')
import numpy as np
import pickle
import torch
import sys 

from conf import settings

from utils import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import svm
import argparse

import copy

import pdb

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_nr",         default=1,      type=int, help="Which run should be taken?")

parser.add_argument("--attack",         default='fgsm',           help=settings.HELP_ATTACK)
parser.add_argument("--detector",       default='InputMFS',       help=settings.HELP_DETECTOR)
parser.add_argument("--net",            default='cif10',          help=settings.HELP_NET)
parser.add_argument("--mode",           default='test',           help="Choose test or validation case")
parser.add_argument("--nr",             default='-1',   type=int, help=settings.HELP_LAYER_NR)
parser.add_argument("--wanted_samples", default='1500', type=int, help=settings.HELP_WANTED_SAMPLES)
parser.add_argument("--clf",            default='LR',             help="Logistic Regression (LR) or Random Forest (RF)")
parser.add_argument("--num_classes",    default='10',   type=int, help=settings.HELP_NUM_CLASSES)

# parser.add_argument("--eps",       default='-1',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
parser.add_argument("--eps",       default='8./255.',            help=settings.HELP_AA_EPSILONS)
# parser.add_argument("--eps",       default='4./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",       default='2./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",       default='1./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",       default='1./255.',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps",       default='0.5/255.',      help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")

args = parser.parse_args()

# output data
output_path_dir = create_dir_detection(args, root='./data/detection/')
save_args_to_file(args, output_path_dir)
logger = Logger(output_path_dir + os.sep + 'log.txt')
log_header(logger, args, output_path_dir, sys) # './data/extracted_characteristics/imagenet32/wrn_28_10/std/8_255/LayerMFS'

# load characteristics
logger.log('INFO: Loading characteristics...')
layer_name = layer_name_cif10
if args.net == 'cif10vgg':
    layer_name = layer_name_cif10vgg


# input data
extracted_characteristics_path = create_dir_extracted_characteristics(args, root='./data/extracted_characteristics/', wait_input=False)
characteristics_path, characteristics_advs_path = create_save_dir_path(extracted_characteristics_path, args, filename='characteristics' )

logger.log("characteristics_path:      " + str(characteristics_path) )
logger.log("characteristics_advs_path: " + str(characteristics_advs_path) )

characteristics =     torch.load(characteristics_path)[:args.wanted_samples]
characteristics_adv = torch.load(characteristics_advs_path)[:args.wanted_samples]

shape = np.shape(characteristics)
logger.log("shape: " + str(shape))

if shape[0] < args.wanted_samples:
    logger.log("CAUTION: The actual number is smaller as the wanted samples!")

k = shape[0]

test_size = 0.2
adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_test = train_test_split(characteristics_adv, np.ones(k), test_size=test_size, random_state=42)
b_X_train_val, b_X_test, b_y_train_val, b_y_test         = train_test_split(characteristics, np.zeros(k), test_size=test_size, random_state=42)
adv_X_train, adv_X_val, adv_y_train, adv_y_val           = train_test_split(adv_X_train_val, adv_y_train_val, test_size=test_size, random_state=42)
b_X_train, b_X_val, b_y_train, b_y_val                   = train_test_split(b_X_train_val, b_y_train_val, test_size=test_size, random_state=42)

X_train = np.concatenate(( b_X_train, adv_X_train) )
y_train = np.concatenate(( b_y_train, adv_y_train) )

if args.mode == 'test':
    X_test = np.concatenate( (b_X_test, adv_X_test) )
    y_test = np.concatenate( (b_y_test, adv_y_test) )
elif args.mode == 'validation':
    X_test = np.concatenate( (b_X_val, adv_X_val) )
    y_test = np.concatenate( (b_y_val, adv_y_val) )
else:
    logger.log('Not a valid mode')

logger.log("b_X_train" + str(b_X_train.shape) )
logger.log("adv_X_train" + str(adv_X_train.shape) )

logger.log("b_X_test" + str(b_X_test.shape) )
logger.log("adv_X_test" + str(adv_X_test.shape) )

#train classifier
logger.log('Training classifier...')


#special case
# if (detector == 'LayerMFS'or detector =='LayerPFS') and net == 'imagenet33' and (attack_method=='std' or attack_method=='cw' or attack_method=='df'):
#     print("SVM")
#     # from cuml.svm import SVC
#     scaler  = MinMaxScaler().fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test  = scaler.transform(X_test)
#     if detector == 'LayerMFS':
#         gamma = 0.1
#         if attack_method == 'cw':
#             C=1
#         else:
#             C=10
#     else:
#         C=10
#         gamma = 0.01
#     # clf = SVC(probability=True, C=C, gamma=gamma)
#     clf = svm.SVC(probability=True, C=C, gamma=gamma ) # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# else:
#     # clf = RandomForestClassifier(max_depth=3, n_estimators=300)


# C_list = [1, 5, 10, 15, 20]
# gamma_list = [0.001, 0.01, 0.1, 1]

# clf = LogisticRegression()

# C_list = [1, 5, 10]
# gamma_list = [0.01, 0.1]

# C = 1
# gamma = 0.01
# clf = svm.SVC(probability=True, C=C, gamma=gamma ) 


# for c in C_list:
#     for g in gamma_list:
#         clf.set_params(C=c, gamma=g )
#         clf.fit(X_train,y_train)
#         print ("C ", c, " gamma", g, "train error: ", clf.score(X_train, y_train) )
#         print ("C ", c, " gamma", g, "test error:  ", clf.score(X_test, y_test) )


if args.clf == 'LR' and settings.SAVE_CLASSIFIER:
    clf = LogisticRegression()
    logger.log(clf)
    clf.fit(X_train,y_train)
    logger.log("train error: " + str(clf.score(X_train, y_train)) )
    logger.log("test error:  " + str(clf.score(X_test,  y_test)) )

if args.clf == 'RF' and not settings.SAVE_CLASSIFIER:
    # trees = [100, 200, 300, 400, 500]
    # trees = [600, 700, 800, 900]
    # trees = [ 500 ]
    trees = [ 300 ]

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    save_clf = copy.deepcopy(clf)
    test_score_save = 0

    for tr in trees:
        clf.set_params(n_estimators=tr)
        clf.fit(X_train, y_train)

        test_score = clf.score(X_test, y_test)
        logger.log("Tr "+ str(tr) + "train error: " + str(clf.score(X_train, y_train)) )
        logger.log("Tr "+ str(tr) + "test error:  " + str(test_score) )
        if test_score > test_score_save:
            save_clf = copy.deepcopy(clf)
    clf = copy.deepcopy(save_clf)


#save classifier
classifier_pth = output_path_dir + os.sep + str(args.clf) + '.clf'
if settings.SAVE_CLASSIFIER:
    torch.save(clf, classifier_pth)
else:
    clf = torch.load(classifier_pth)


logger.log('Evaluating classifier...')
y_hat =    clf.predict(X_test)
y_hat_pr = clf.predict_proba(X_test)[:, 1]

# logger.log( "train error: " + str(clf.score(X_train, y_train)) )
# logger.log( "test error:  " + str(clf.score(X_test, y_test)) )

nr_not_detect_adv = 0

benign_rate = 0
benign_guesses = 0
ad_guesses = 0
ad_rate = 0
for i in range(len(y_hat)):
    if y_hat[i] == 0:
        benign_guesses +=1
        if y_test[i]==0:
            benign_rate +=1
    else:
        ad_guesses +=1
        if y_test[i]==1:
            ad_rate +=1

    if y_test[i] == 1:
        if y_hat[i] == 0:
            nr_not_detect_adv  +=1

acc = (benign_rate+ad_rate)/len(y_hat)        
TP = 2*ad_rate/len(y_hat)
TN = 2*benign_rate/len(y_hat)

precision = ad_rate/ad_guesses

TPR = 2 * ad_rate / len(y_hat)
recall = round(100*TPR, 2)

prec = precision 
rec = TPR 


auc = round(100*roc_auc_score(y_test, y_hat_pr), 2)
acc = round(100*acc, 2)
pre = round(100*precision, 1)
tpr = round(100*TP, 2)
f1  = round((2 * (prec*rec) / (prec+rec))*100, 2)
fnr = round(100 - tpr, 2)



logger.log('F1-Measure: ' + str(f1) )
logger.log('PREC: ' + str(pre) )
logger.log('ACC: ' + str(acc) )
logger.log('AUC: ' + str(auc) )
logger.log('TNR: ' + str(round(100*TN, 2)) ) # True negative rate/normal detetcion rate/selectivity is 
logger.log('TPR: ' + str(tpr) )# True positive rate/adversarial detetcion rate/recall/sensitivity is 
logger.log('FNR: ' + str(fnr) )
logger.log('RES:, AUC, ACC, PRE, TPR, F1, FNR' )
logger.log('RES:,' + str(auc) + ',' + str(acc) + ',' + str(pre) + ',' + str(tpr) + ',' + str(f1) + ',' + str(fnr) )
logger.log('<==========================================================================')

# tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=list(range(args.num_classes))).ravel()
# tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=list(range(args.num_classes))).ravel()

aa, bb, cc, dd = perf_measure(y_test, y_hat)

fpr, tpr, _ = roc_curve(y_test, y_hat_pr)
print("fpr", fpr)
print("tpr", tpr)
