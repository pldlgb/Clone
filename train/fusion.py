from train.gnn import batch_gat_loss
from train.conv import train_conv
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import math
from torch.autograd import Variable
from torch.nn.modules import module

from utils.utils import save_model, CUDA
from model.models import SpKBGATModified, SpKBGATConvOnly
from train.gnn import batch_gat_loss

def fusionKGC(args, Corpus, entity_embedding, relation_embedding, node_neighbors_2hop = None):
    # Build a Fusion KGC model
    # also need fusion pattern

    print("------------This is FusionKGC training------- ")
    print("Defining model")
    print("---------------GAT---------------")
    model_gat = SpKBGATModified(entity_embedding,relation_embedding,
                                args.entity_out_dim,args.relation_out_dim,args.drop_GAT,args.alpha,args.nheads_GAT)
    print("---------------Conv---------------")
    model_conv = SpKBGATConvOnly(entity_embedding, relation_embedding, args.entity_out_dim, args.entity_out_dim,
                                args.drop_conv, args.alpha_conv, args.nheads_GAT, args.out_channels)
    
    if CUDA:
        model_gat.cuda()
        model_conv.cuda()
    
    ### Define GAT training parameter
    optim_gat = torch.optim.Adam(model_gat.parameters(), lr=args.lr, weight_decay = args.weight_decay_gat)
    scheduler_gat = torch.optim.lr_scheduler.StepLR(optim_gat,step_size=500,gamma=0.5,last_epoch=-1)
    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)
    current_batch_2hop_indices = torch.LongTensor([[0,0,0,0]])
    num_iters_gat = math.ceil(len(Corpus.train_indices)/args.batch_size_gat)
    if args.use_2hop:
        current_batch_2hop_indices = Corpus.get_batch_nhop_neighbors_all(args,
                                        Corpus.unique_entities_train,node_neighbors_2hop)
    
    ### Define Conv training parameter
    optim_conv = torch.optim.Adam(model_conv.parameters(),lr=args.lr,weight_decay=args.weight_decay_conv)
    scheduler_conv = torch.optim.lr_scheduler.StepLR(optim_conv, step_size=25, gamma=0.5, last_epoch=-1)
    margin_loss = torch.nn.SoftMarginLoss()
    num_iters_conv = math.ceil(len(Corpus.train_indices)/args.batch_size_conv)

    ### gat training 
    epoch_gat_losses = []   # losses of all epochs
    epoch_conv_losses = []
    print("Number of epochs {}".format(args.epochs_gat))

    tmp_epoches=100

    for epoch in range(tmp_epoches):
        print("\nepoch->",epoch)
        random.shuffle(Corpus.train_triples)
        Corpus.train_indices = np.array(list(Corpus.train_triples)).astype(np.int32)

        model_gat.train()
        model_conv.train()
        Corpus.batch_size = args.batch_size_gat
        Corpus.invalid_valid_ratio = int(args.valid_invalid_ratio_gat)
        for _ in range(10):
            start_time = time.time()
            epoch_gat_loss = []

            for iter in range(num_iters_gat):
                start_time_iter=time.time()
                train_gat_indices,train_gat_values = Corpus.get_iteration_batch(iter)

                if CUDA:
                    train_gat_indices = Variable(
                        torch.LongTensor(train_gat_indices)).cuda()
                    train_gat_values = Variable(torch.FloatTensor(train_gat_values)).cuda()

                else:
                    train_gat_indices = Variable(torch.LongTensor(train_gat_indices))
                    train_gat_values = Variable(torch.FloatTensor(train_gat_values))
                
                entity_embed, relation_embed = model_gat(Corpus,Corpus.train_adj_matrix,
                                                        train_gat_indices,current_batch_2hop_indices,
                                                        model_conv.final_entity_embeddings,
                                                        model_conv.final_relation_embeddings)
                optim_gat.zero_grad()
                loss_gat = batch_gat_loss(gat_loss_func, train_gat_indices,entity_embed,
                                    relation_embed,args.valid_invalid_ratio_gat)
                
                loss_gat.backward()
                optim_gat.step()

                epoch_gat_loss.append(loss_gat.data.item())

                end_time_iter = time.time()

                print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                    iter, end_time_iter - start_time_iter, loss_gat.data.item()))

            scheduler_gat.step()
            print("Epoch {} , average loss {} , epoch_time {}".format(
                epoch, sum(epoch_gat_loss) / len(epoch_gat_loss), time.time() - start_time))
            epoch_gat_losses.append(sum(epoch_gat_loss) / len(epoch_gat_loss))
        save_model(model_gat, args.data, epoch, args.output_folder)

        
        # model_conv.final_entity_embeddings = nn.Parameter(F.normalize(model_conv.final_entity_embeddings+model_gat.final_entity_embeddings,p=2, dim=1))
        # model_gat.final_relation_embeddings = nn.Parameter(F.normalize(model_conv.final_relation_embeddings+model_gat.final_relation_embeddings,p=2,dim=1))

        ### 不知道会不会影响到之前的网络
        epoch_conv_loss = []
        Corpus.batch_size = args.batch_size_conv
        Corpus.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)
        for iter in range(num_iters_conv):
            start_time_iter = time.time()
            train_conv_indices, train_conv_values = Corpus.get_iteration_batch(iter)

            if CUDA:
                train_conv_indices = Variable(
                    torch.LongTensor(train_conv_indices)).cuda()
                train_conv_values = Variable(torch.FloatTensor(train_conv_values)).cuda()

            else:
                train_conv_indices = Variable(torch.LongTensor(train_conv_indices))
                train_conv_values = Variable(torch.FloatTensor(train_conv_values))

            # preds = model_conv(Corpus, Corpus.train_adj_matrix, train_conv_indices)
            preds = model_conv(train_conv_indices,model_gat.final_entity_embeddings,model_gat.final_relation_embeddings)
            optim_conv.zero_grad()

            loss_conv = margin_loss(preds.view(-1), train_conv_values.view(-1))

            loss_conv.backward()
            optim_conv.step()

            epoch_conv_loss.append(loss_conv.data.item())

            end_time_iter = time.time()

            # print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
            #     iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler_conv.step()
        print("------------Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_conv_loss) / len(epoch_conv_loss), time.time() - start_time))
        epoch_conv_losses.append(sum(epoch_conv_loss) / len(epoch_conv_loss))
        print("---------------------Save Conv----------------------")
        save_model(model_conv, args.data, epoch, args.output_folder + "conv/")
