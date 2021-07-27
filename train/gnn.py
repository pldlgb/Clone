import torch
import torch.nn as nn
import numpy as np
import random
import time
from torch.autograd import Variable

from model.models import SpKBGATModified
from utils.utils import save_model, CUDA

# gat_loss_func = nn.MarginRankingLoss(margin=0.5)

def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed, valid_invalid_ratio_gat):
    '''
    parameters:
        gat_loss_func : loss function
        train_indices : triples list with pos and neg
        entity_embed  : entity_embedding
        relation_embed: relation_embedding
        valid_invalid_ratio_gat 
    
    return :
        loss 
    '''
    len_pos_triples = int(
        train_indices.shape[0] / (int(valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss

def train_gat(args, Corpus, entity_embeddings, relation_embeddings, node_neighbors_2hop = None):

    # Creating the gat model here.
    ####################################

    print("Defining model")

    print("\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
    
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)

    if CUDA:
        model_gat.cuda()

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)
    # water
    current_batch_2hop_indices = torch.LongTensor([[0,0,0,0]])
    # current_batch_2hop_indices = torch.tensor([])
    if(args.use_2hop):
        current_batch_2hop_indices = Corpus.get_batch_nhop_neighbors_all(args,
                                                                          Corpus.unique_entities_train, node_neighbors_2hop)
    # water
    # if CUDA:
    #     current_batch_2hop_indices = Variable(
    #         torch.LongTensor(current_batch_2hop_indices)).cuda()
    # else:
    #     current_batch_2hop_indices = Variable(
    #         torch.LongTensor(current_batch_2hop_indices))

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus.train_triples)
        Corpus.train_indices = np.array(
            list(Corpus.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(Corpus.train_indices) // args.batch_size_gat) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed = model_gat(
                Corpus, Corpus.train_adj_matrix, train_indices, current_batch_2hop_indices)

            optimizer.zero_grad()

            loss = batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed,args.valid_invalid_ratio_gat)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

    save_model(model_gat, args.data, epoch, args.output_folder)

