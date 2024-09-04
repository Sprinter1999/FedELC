#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import random
import time
from datetime import datetime
import os

import torchvision
import torch
from torch.utils.data import DataLoader

from utils import load_dataset
from utils.options import args_parser
from utils.sampling import sample_iid, sample_noniid_shard, sample_dirichlet
from utils.utils import noisify_label

from fl_models.fed import LocalModelWeights
from fl_models.nets import get_model
from fl_models.test import test_img
from fl_models.update import get_local_update_objects

from resnets.build_model import build_model

if __name__ == '__main__':

    start = time.time()
    # parse args
    args = args_parser()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu',
    )

    args.all_clients = False
    args.schedule = [int(x) for x in args.schedule]

    # FIXME: Please dismiss these methods
    args.send_2_models = args.method in [
        'coteaching', 'coteaching+', 'dividemix', ]

    # Seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    # The number of selected clients in each global round
    args.selected_total_clients_num = args.num_users * args.frac



    # FIXME: Arbitrary gaussian noise, not used in FedELC but used in FedRN, please dismiss it
    gaussian_noise = torch.randn(1, 3, 32, 32)
    if(args.dataset == 'clothing1m'):
        gaussian_noise = torch.randn(1, 3, 224, 224)





    ##############################
    # Load dataset and split users
    ##############################
    dataset_train, dataset_test, args.num_classes = load_dataset(args.dataset)
    labels = np.array(dataset_train.train_labels)
    labels_torch = torch.tensor(labels)
    img_size = dataset_train[0][0].shape  # used to get model
    args.img_size = int(img_size[1])






    #TODO: varibles for the design of the fedELC method
    num_total_samples = len(labels)
    # global Soft_labels
    # forms a N * 100 or 100 classes all-zero matrix and moves it to GPU
    Soft_labels = torch.zeros([num_total_samples, args.num_classes], dtype=torch.float)
    # global Soft_labels_flag
    Soft_labels_flag = torch.zeros([num_total_samples], dtype=torch.int)
    # Soft_labels.cuda()



    True_Labels = copy.deepcopy(labels_torch)
    args.True_Labels = True_Labels
    args.Soft_labels = Soft_labels

    
    # for fedrn and some other baselines...
    args.warmup_epochs = int(0.2 * args.epochs)




    # Sample users (iid / non-iid)
    if args.partition == 'shard':  # non-iid
        if(args.dataset == 'cifar10'):
            # 5 classes for a client at most (total clients=100)
            args.num_shards = 500
        elif(args.dataset == 'cifar100'):
            # 20 classes for a client at most (total clients=100)
            args.num_shards = 2000

        print("[Partitioning Via Sharding....]")
        dict_users = sample_noniid_shard(
            labels=labels,
            num_users=args.num_users,
            num_shards=args.num_shards,
        )

    elif args.partition == 'dirichlet':
        print("[Partitioning Via Dir....]")
        dict_users = sample_dirichlet(
            labels=labels,
            num_clients=args.num_users,
            alpha=args.dd_alpha,
            num_classes=args.num_classes,
        )












    print("#############   Print  all  args param. ##########")
    for x in vars(args).items():
        print(x)

    if not torch.cuda.is_available():
        exit('ERROR: Cuda is not available!')
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)





    print("########################## Add Label Noise #################################")

    client_noise_map= {}

    ##############################
    # Add label noise to data
    ##############################
    if sum(args.noise_group_num) != args.num_users:
        exit('Error: sum of the number of noise group have to be equal the number of users')

    if len(args.group_noise_rate) == 1:
        args.group_noise_rate = args.group_noise_rate * 2

    if not len(args.noise_group_num) == len(args.group_noise_rate) and \
            len(args.group_noise_rate) * 2 == len(args.noise_type_lst):
        exit('Error: The noise input is invalid.')

    args.group_noise_rate = [(args.group_noise_rate[i * 2], args.group_noise_rate[i * 2 + 1])
                             for i in range(len(args.group_noise_rate) // 2)]

    user_noise_type_rates = []
    for num_users_in_group, noise_type, (min_group_noise_rate, max_group_noise_rate) in zip(
            args.noise_group_num, args.noise_type_lst, args.group_noise_rate):
        noise_types = [noise_type] * num_users_in_group

        step = (max_group_noise_rate - min_group_noise_rate) / \
            num_users_in_group
        noise_rates = np.array(range(num_users_in_group)) * \
            step + min_group_noise_rate

        user_noise_type_rates += [*zip(noise_types, noise_rates)]

    for user, (user_noise_type, user_noise_rate) in enumerate(user_noise_type_rates):
        if user_noise_type != "clean":
            data_indices = list(copy.deepcopy(dict_users[user]))

            client_noise_map[user] = user_noise_rate

            # for reproduction
            random.seed(args.seed)
            random.shuffle(data_indices)

            noise_index = int(len(data_indices) * user_noise_rate)

            for d_idx in data_indices[:noise_index]:
                true_label = dataset_train.train_labels[d_idx]
                noisy_label = noisify_label(
                    true_label, num_classes=args.num_classes, noise_type=user_noise_type)
                dataset_train.train_labels[d_idx] = noisy_label






    #############################################################################
    ############################## Log key metrics ##############################
    #############################################################################


    logging_args = dict(
        batch_size=args.test_bs,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    acc_list_glob1 = []
    test_loss_list_glob1 = []
    precision_list_glob1 = []
    recall_list_glob1 = []
    f1_list_glob1 = []

    percentage_correct_glob1 = []
    before_percentage_correct_glob1, after_percentage_correct_glob1, merged_percentage_correct_glob1 = [], [], []



    # log_train_data_loader = torch.utils.data.DataLoader(
    #     dataset_train, **logging_args)

    log_test_data_loader = torch.utils.data.DataLoader(
        dataset_test, **logging_args)






    ##############################
    # Build model
    ##############################
    net_glob = build_model(args)
    # net_glob = net_glob.to(args.device)



    if args.model == 'resnet50':
        args.feature_dim = 2048
    elif args.model == 'resnet34':
        args.feature_dim = 512
    elif args.model == 'resnet18':
        args.feature_dim = 512
    else:
        args.feature_dim = 128






    ##############################
    # Training
    ##############################
    CosineSimilarity = torch.nn.CosineSimilarity()
    # base_optim = torch.optim.SGD
    # sam_optimizer = SAM(net_glob.parameters(), base_optim, rho=args.fedsam_rho, adaptive=False,
    #                     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ##############################
    # Class centroids for robust FL
    ##############################
    # glob_centroid = {i: None for i in range(args.num_classes)}
    # f_G &  Used for other baseline, please dismiss
    #FIXME: used in RobustFL (IEEE Intelligent Systems 2022)
    f_G = torch.randn(args.num_classes, args.feature_dim, device=args.device)
    forget_rate_schedule = []

    pred_user_noise_rates = [args.forget_rate] * args.num_users







    # Initialize local model weights
    fed_args = dict(
        all_clients=args.all_clients,
        num_users=args.num_users,
        method=args.method,
        dict_users=dict_users,
        args=args,
    )




    local_weights = LocalModelWeights(net_glob=net_glob, **fed_args)
    if args.send_2_models:
        local_weights2 = LocalModelWeights(net_glob=net_glob2, **fed_args)



    ######################################################################
    ###################### Initialize local update objects ###############
    ######################################################################

    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        noise_rates=pred_user_noise_rates,
        gaussian_noise=gaussian_noise,
        glob_centroid=f_G,
    )

    for i in range(args.num_users):
        # local = local_update_objects[i]
        # local.weight = copy.deepcopy(net_glob.state_dict())
        local_update_objects[i].weight = copy.deepcopy(net_glob.state_dict())







    #########################################
    ###### Global Training for FNLL #########
    #########################################



    for epoch in range(args.epochs):
        print("\n####################### Global Epoch {} Starts...".format(epoch))

        # FIXME: we do not use learning rate scheduler
        # if (epoch + 1) in args.schedule:
        #     print("Learning Rate Decay Epoch {}".format(epoch + 1))
        #     print("{} => {}".format(args.lr, args.lr * args.lr_decay))
        #     args.lr *= args.lr_decay


        local_losses = []
        local_losses2 = []
        args.g_epoch = epoch
        feature_locals = []
        local_percentages = []
        before_local_percentages, after_local_percentages, merged_local_percentages = [], [], []

        print_flag=False
        

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Update
        # counter = 0
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args
            # percentage_correct = 0
            percentage_correct, before_percentage_correct, after_percentage_correct, merged_percentage_correct = 0, 0, 0, 0



            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"...Select Client {client_num} and actual client idx#{idx} and name of updater {local.update_name} at the time: {current_time}...")

            if args.method == "fedlsr":
                w, loss = local.train(net=copy.deepcopy(
                    net_glob).to(args.device), cur_round=epoch)

            

            elif args.method == "fedELC":
                #TODO: Warm-up epochs
                epoch_of_stage1 = args.epoch_of_stage1 # just set it as the initial paper in the ICH dataset

                if epoch < epoch_of_stage1:
                    local_weights.noisy_clients = 0
                elif epoch == epoch_of_stage1 and client_num==0: # client selection by GMM
                    loader = DataLoader(dataset=dataset_train, batch_size=32,shuffle=False, num_workers=4)
                    criterion = torch.nn.CrossEntropyLoss(reduction='none')
                    from utils.utils import get_output
                    local_output, loss = get_output(loader, net_glob.to(args.device), args, False, criterion)
                    metrics = np.zeros((args.num_users, args.num_classes)).astype("float")
                    num = np.zeros((args.num_users, args.num_classes)).astype("float")
                    for id in range(args.num_users):
                        idxs = dict_users[id]
                        for idxx in idxs:
                            c = dataset_train.train_labels[idxx]
                            num[id, c] += 1
                            metrics[id, c] += loss[idxx]
                    # print("^^^^^^")
                    # print(metrics)
                    metrics = metrics / num 
                    # print("^^^^^^")
                    # print(metrics)
                    for i in range(metrics.shape[0]):
                        for j in range(metrics.shape[1]):
                            if np.isnan(metrics[i, j]):
                                metrics[i, j] = np.nanmin(metrics[:, j])
                    for j in range(metrics.shape[1]):
                        metrics[:, j] = (metrics[:, j]-metrics[:, j].min()) / \
                            (metrics[:, j].max()-metrics[:, j].min())
                        
                    from sklearn.mixture import GaussianMixture
                    vote = []
                    for i in range(9):
                        gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
                        gmm_pred = gmm.predict(metrics)
                        noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
                        noisy_clients = set(list(noisy_clients))
                        vote.append(noisy_clients)
                   
                    cnt = []
                    for i in vote:
                        cnt.append(vote.count(i))
                    noisy_clients = list(vote[cnt.index(max(cnt))])
                    user_id = list(range(args.num_users))
                    clean_clients = list(set(user_id) - set(noisy_clients))
                    
                    local_weights.noisy_clients = noisy_clients
                    local_weights.clean_clients = clean_clients
                    local_weights.client_tag = [] # to indicate if this client is clean (1)

                    # print(f"###############\n Len of NOISY/CLEAN clients: {len(noisy_clients)} VS {len(clean_clients)} \n ###############")


                    noisyclient_rate_list, cleanclient_rate_list = [], []
                    # FOr noisy client, we retreive the noisy rate from client_noise_map
                    for each in noisy_clients:
                        noisyclient_rate_list.append(client_noise_map[each])
                    for each in clean_clients:
                        cleanclient_rate_list.append(client_noise_map[each])

                    print(f"#######[Division]########\n Len of NOISY/CLEAN clients: {len(noisy_clients)} VS {len(clean_clients)} \n ###############")
                    print(f"#######[Division]########\n Sum of noisy clients' rates total quantity: {sum(noisyclient_rate_list)}")
                    print(f"#######[Division]########\n Sum of clean clients' rates total quantity: {sum(cleanclient_rate_list)}")
                
                else:
                    pass

                # local training
                if epoch < epoch_of_stage1: # stage 1, 
                    w, loss = local.train_stage1(net=copy.deepcopy(net_glob).to(args.device))
                else: # stage 2, 
                    def sigmoid_rampup(current, begin, end):
                        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
                        current = np.clip(current, begin, end)
                        phase = 1.0 - (current-begin) / (end-begin)
                        return float(np.exp(-5.0 * phase * phase))

                    def get_current_consistency_weight(rnd, begin, end):
                        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
                        return sigmoid_rampup(rnd, begin, end)
                    
                    weight_kd = get_current_consistency_weight(epoch, epoch_of_stage1, args.epochs) * 0.8


                    print_flag = True

                    if idx in local_weights.clean_clients:
                        w, loss = local.train_stage1(net=copy.deepcopy(net_glob).to(args.device))
                        local_weights.client_tag.append(1)
                    else:
                        w, loss, percentage_correct, before_percentage_correct, after_percentage_correct, merged_percentage_correct = local.train_stage2(net=copy.deepcopy(net_glob).to(args.device), global_net=copy.deepcopy(net_glob).to(args.device), weight_kd=weight_kd)
                        local_weights.client_tag.append(0)


            else:
                # FedAvg branch
                w, loss = local.train(copy.deepcopy(net_glob).to(args.device))




            local_weights.update(idx, w)
            local_losses.append(copy.deepcopy(loss))
            local_percentages.append(percentage_correct)

            # if print_flag:
            before_local_percentages.append(before_percentage_correct)
            after_local_percentages.append(after_percentage_correct)

            #TODO: we use merged predictions to finally refine the local labels of the noisy clients
            merged_local_percentages.append(merged_percentage_correct)


        ##############################  
        print(f"Global Epoch {epoch} Local Training is done!.....")


        if local_percentages is not []:
            selected_entries = [i for i in range(len(local_percentages)) if local_percentages[i] != 0]

            local_percentages = [local_percentages[i] for i in selected_entries]

            if len(local_percentages) > 0:
                correct_percentages_avg = sum(local_percentages) / len(local_percentages)
                print(f"[ELC ESTIMATE] Average correct estimate percentage is {correct_percentages_avg}, overall correction rate is: {local_percentages}")
                percentage_correct_glob1.append(correct_percentages_avg)


            else:
                percentage_correct_glob1.append(0)
        else:
            percentage_correct_glob1.append(0)



        if before_local_percentages is not []:
            selected_entries = [i for i in range(len(before_local_percentages)) if before_local_percentages[i] != 0]

            before_local_percentages = [before_local_percentages[i] for i in selected_entries]

            if len(before_local_percentages) > 0:
                before_correct_percentages_avg = sum(before_local_percentages) / len(before_local_percentages)
                print(f"[Before training: PREDICT] Average correct prediction percentage before is {before_correct_percentages_avg}, overall prediction rate is: {before_local_percentages}")
                before_percentage_correct_glob1.append(before_correct_percentages_avg)


            else:
                before_percentage_correct_glob1.append(0)
        
        if after_local_percentages is not []:
            selected_entries = [i for i in range(len(after_local_percentages)) if after_local_percentages[i] != 0]

            after_local_percentages = [after_local_percentages[i] for i in selected_entries]

            if len(after_local_percentages) > 0:
                after_correct_percentages_avg = sum(after_local_percentages) / len(after_local_percentages)
                print(f"[After training: PREDICT] Average correct prediction percentage after is {after_correct_percentages_avg}, overall prediction rate is: {after_local_percentages}")
                after_percentage_correct_glob1.append(after_correct_percentages_avg)


            else:
                after_percentage_correct_glob1.append(0)
        
        if merged_local_percentages is not []:
            selected_entries = [i for i in range(len(merged_local_percentages)) if merged_local_percentages[i] != 0]

            merged_local_percentages = [merged_local_percentages[i] for i in selected_entries]

            if len(merged_local_percentages) > 0:
                merged_correct_percentages_avg = sum(merged_local_percentages) / len(merged_local_percentages)
                print(f"[Merged training: PREDICT] Average correct prediction percentage after is {merged_correct_percentages_avg}, overall prediction rate is: {merged_local_percentages}")
                merged_percentage_correct_glob1.append(merged_correct_percentages_avg)


            else:
                merged_percentage_correct_glob1.append(0)





        # show local loss mean
        print(f"[LOSS] Average local loss mean is {sum(local_losses) / len(local_losses)}")

        #TODO: update the global model weights
        w_glob = local_weights.average()  # update global weights


    


        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f'\n#### Testing for Method {args.method} in Round {epoch} and current time is {current_time} ####')
        net_glob.to(args.device)

        # update global weights
        local_weights.global_w_init = copy.deepcopy(net_glob.state_dict())

        local_weights.init()  # clear temp local weights for the next round aggregation

        # train_acc, train_loss = test_img(net_glob, log_train_data_loader, args)
        accuracy, test_loss, precision, recall, f1 = test_img(
            net_glob, log_test_data_loader, args)
        # for logging purposes
        # results = dict(train_acc=train_acc, train_loss=train_loss,
        #                test_acc=test_acc, test_loss=test_loss, )
        # results = dict(accuracy=accuracy, test_loss=test_loss, precision=precision, recall=recall, f1=f1)

        acc_list_glob1.append(accuracy)
        test_loss_list_glob1.append(test_loss)
        precision_list_glob1.append(precision)
        recall_list_glob1.append(recall)
        f1_list_glob1.append(f1)


    print("####################### Global Model 1")
    print(f"Total test acc list for global model 1 is shown: {acc_list_glob1}")
    print("#############################################")
    print(
        f"Total test loss list for global model 1 is shown: {test_loss_list_glob1}")
    print("#############################################")
    print(
        f"Total precision list for global model 1 is shown: {precision_list_glob1}")
    print("#############################################")
    print(
        f"Total recall list for global model 1 is shown: {recall_list_glob1}")
    print("#############################################")
    print(f"Total f1 list for global model 1 is shown: {f1_list_glob1}")

    print("@@@@@@@@ Print Average Metrics of last 10 rounds for global model 1 @@@@@@@@@")
    print(
        f"Mean Accuracy of last 10 rounds for global model 1 is shown: {np.mean(acc_list_glob1[-10:])}")
    print(
        f"Mean F1 Score of last 10 rounds for global model 1 is shown: {np.mean(f1_list_glob1[-10:])}")
    print(
        f"Mean Precision of last 10 rounds for global model 1 is shown: {np.mean(precision_list_glob1[-10:])}")
    print(
        f"Mean Recall of last 10 rounds for global model 1 is shown: {np.mean(recall_list_glob1[-10:])}")
    print(
        f"Mean Test Loss of last 10 rounds for global model 1 is shown: {np.mean(test_loss_list_glob1[-10:])}")
    print("#############################################")
    #percentage_correct_glob1
    print(
        f"[Estimate] Overall Estimation Percentage Correct  is shown: {percentage_correct_glob1}")
    
    print(
        f"[Before Predict] Overall Before Prediction Percentage Correct  is shown: {before_percentage_correct_glob1}")
    print(
        f"[After Predict] Overall After Prediction Percentage Correct  is shown: {after_percentage_correct_glob1}")
    print(
        f"[Merged Predict] Overall Merged Prediction Percentage Correct  is shown: {merged_percentage_correct_glob1}")
    print("#############################################")