#!/usr/bin/env python3
""" AutoAttack Foolbox

author Peter Lorenz
"""

print('Load modules...')
import os, sys
import argparse
import pdb
import torch

import numpy as np
from tqdm import tqdm

from conf import settings

import foolbox
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import  L2DeepFoolAttack, LinfBasicIterativeAttack, FGSM, L2CarliniWagnerAttack, FGM, PGD

from utils import *


if __name__ == '__main__':
    #processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_nr",         default=1,    type=int, help="Which run should be taken?")

    parser.add_argument("--attack",         default='fgsm',           help=settings.HELP_ATTACK)
    parser.add_argument("--net",            default='cif10',          help=settings.HELP_NET)
    parser.add_argument("--img_size",       default='32',   type=int, help=settings.HELP_IMG_SIZE)
    parser.add_argument("--num_classes",    default='10',   type=int, help=settings.HELP_NUM_CLASSES)
    parser.add_argument("--wanted_samples", default='2000', type=int, help=settings.HELP_WANTED_SAMPLES)
    parser.add_argument("--all_samples",    default='4000', type=int, help="Samples from generate Clean data")


    # Only for Autoatack
    parser.add_argument('--norm',       type=str, default='Linf')
    parser.add_argument('--eps',        type=str, default='8./255.')
    parser.add_argument('--individual',            action='store_true')
    parser.add_argument('--batch_size', type=int, default=1500)
    parser.add_argument('--log_path',   type=str, default='log.txt')
    parser.add_argument('--version',    type=str, default='standard')

    parser.add_argument('--net_normalization', action='store_false', help=settings.HELP_NET_NORMALIZATION)
    

    args = parser.parse_args()

    # output data
    output_path_dir = create_dir_attacks(args, root='./data/attacks/')

    save_args_to_file(args, output_path_dir)
    logger = Logger(output_path_dir + os.sep + 'log.txt')
    log_header(logger, args, output_path_dir, sys) # './data/attacks/imagenet32/wrn_28_10/fgsm'

    #load model
    logger.log('INFO: Load model...')
    model, preprocessing = load_model(args)

    model = model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #load correctly classified data
    batch_size = 128 
    if args.net == 'imagenet128' or args.net == 'celebaHQ128':
        batch_size = 64
    elif args.net == 'celebaHQ256':
        batch_size = 24
    elif args.net == 'imagenet':
        batch_size = 32

    # input data    
    clean_data_path = create_dir_clean_data(args, root='./data/clean_data/')
    logger.log('INFO: clean data path: ' + clean_data_path)

    # set up final lists
    images = []
    images_advs = []
    
    success_counter = 0

    counter = 0
    success = []
    success_rate = 0
    logger.log('INFO: Perform attacks...')

    if args.attack == 'std' or args.attack == 'apgd-ce' or args.attack == 'apgd-t' or args.attack == 'fab-t' or args.attack == 'square':
        logger.log('INFO: Load data...')
        testset = load_test_set(args)

        sys.path.append("./submodules/autoattack")
        from submodules.autoattack.autoattack import AutoAttack as AutoAttack_mod
        
        adversary = AutoAttack_mod(model, norm=args.norm, eps=epsilon_to_float(args.eps), log_path=output_path_dir + os.sep + 'log.txt', version=args.version)

        # run attack and save images
        with torch.no_grad():
            if not args.individual:
                logger.log("INFO: mode: std; not individual")
                # raise NotImplementedError("mode: std; not individual")

                for x_test, y_test in testset:
                    adv_complete, max_nr = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)

                    tmp_images_advs = []
                    for it, img in enumerate(adv_complete):
                        if not (np.abs(x_test[it] - img) <= 1e-5).all():
                            images.append(x_test[it])
                            tmp_images_advs.append(img)
                            success_counter = success_counter + 1
                            if (success_counter % 1000) == 0:
                                get_debug_info( msg="success_counter " + str(success_counter) + " / " + str(args.wanted_samples) )

                    success.append( len(tmp_images_advs) / max_nr )

                    images_advs += tmp_images_advs 
                    tmp_images_advs = []

                    success_rate  = np.mean(success)
                    if success_counter >= args.wanted_samples:
                        print( " success: {:2f}".format(success_rate) )
                        break
            else:
                logger.log("ERR: not implemented yet!")
                raise NotImplementedError("ERR: not implemented yet!")


    elif args.attack == 'fgsm' or args.attack == 'bim' or args.attack == 'pgd' or args.attack == 'df' or args.attack == 'cw': 

        testset = torch.load(clean_data_path + os.sep + 'clean_data')[:args.all_samples]
        logger.log("INFO: len(testset): {}".format(len(testset)))
        
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # total_len = len(testset) / batch_size

        #setup depending on attack
        if args.attack == 'fgsm':
            attack = FGSM()
            epsilons = [0.03] 
        elif args.attack == 'bim':
            attack = LinfBasicIterativeAttack()
            epsilons = [0.03]
        elif args.attack == 'pgd':
            attack = PGD()
            epsilons = [0.03]
        elif args.attack == 'df':
            attack = L2DeepFoolAttack()
            epsilons = None
        elif args.attack == 'cw':
            attack = L2CarliniWagnerAttack(steps=1000)
            epsilons = None
        elif args.attack == 'autoattack':
            logger.log("Err: Auttoattack is started from another script! attacks_autoattack.py")
            raise NotImplementedError("Err: Wrong Keyword use 'std' for 'ind' for AutoAttack!")
        else:
            logger.log('Err: unknown attack')
            raise NotImplementedError('Err: unknown attack')


        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

        logger.log('eps: {}'.format(epsilons))

        multipl = 1 # 0.25
        # stop_round = round(total_len * 1)
        stop_round = args.wanted_samples


        for image, label in test_loader:

            # if counter > stop_round:
            #     break
            # counter = counter + 1

            image = torch.squeeze(image)
            label = torch.squeeze(label)

            if batch_size == 1:
                image = torch.unsqueeze(image, 0)
                label = torch.unsqueeze(label, 0)

            image = image.cuda()
            label = label.cuda()

            _, adv, success = attack(fmodel, image, criterion=foolbox.criteria.Misclassification(label), epsilons=epsilons)

            if not (args.attack == 'cw' or args.attack == 'df'):
                adv = adv[0] # list to tensor
                success = success[0]
            for idx, suc in enumerate(success):
                counter = counter + 1
                if suc:
                    images_advs.append(adv[idx].squeeze_(0))
                    images.append(image[idx].squeeze_(0))
                    success_counter = success_counter + 1

                # import pdb; pdb.set_trace()

            if success_counter >= args.wanted_samples:
                logger.log("INFO: wanted samples reached {}".format(args.wanted_samples))
                break

    logger.log("INFO: len(testset):   {}".format( len(testset) ))
    logger.log("INFO: success_counter {}".format(success_counter))
    logger.log("INFO: images {}".format(len(images)))
    logger.log("INFO: images_advs {}".format(len(images_advs)))

    if args.attack == 'std' or args.individual:
        logger.log('INFO: attack success rate: {}'.format(success_rate) )
    else:
        logger.log('INFO: attack success rate: {}'.format(success_counter / counter ) )
    # logger.log('attack success rate: {}'.format(success_counter / len(data_loader.dataset)) )
    # logger.log('attack success rate: {}'.format(success_counter / ((len(data_loader.dataset) - (len(data_loader.dataset) % batch_size)))) )


    # create save dir 
    images_path, images_advs_path = create_save_dir_path(output_path_dir, args)
    logger.log('images_path: ' + images_path)

    torch.save(images,      images_path, pickle_protocol=4)
    torch.save(images_advs, images_advs_path, pickle_protocol=4)

    logger.log('INFO: Done performing attacks and adversarial examples are saved!')