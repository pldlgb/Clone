import torch
import random
import numpy as np
import time
from torch.autograd import Variable

from utils.utils import save_model, CUDA
from model.models import SpKBGATModified, SpKBGATConvOnly

def train_conv(args, Corpus, entity_embeddings, relation_embeddings):

    # Creating convolution model here.
    ####################################

    print("Defining model")
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)
    print("Only Conv model trained")
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_conv, args.alpha_conv,
                                args.nheads_GAT, args.out_channels)
    # water
    # model_conv.load_state_dict(torch.load(
    #     '{0}conv/{1}.pth'.format(args.output_folder, 99)), strict=False)

    if CUDA:
        model_conv.cuda()
        model_gat.cuda()

    model_gat.load_state_dict(torch.load(
        '{}/trained_{}.pth'.format(args.output_folder, args.epochs_gat - 1)), strict=False)
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings
    # water

    Corpus.batch_size = args.batch_size_conv
    Corpus.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus.train_triples)
        Corpus.train_indices = np.array(
            list(Corpus.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus.train_indices) // args.batch_size_conv) + 1

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

            # preds = model_conv(
            #     Corpus, Corpus.train_adj_matrix, train_indices)
            preds = model_conv(train_indices)
            optimizer.zero_grad()

            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            # print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
            #     iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        save_model(model_conv, args.data, epoch,
                   args.output_folder + "conv/")
