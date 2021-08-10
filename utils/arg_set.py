import argparse
import os

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/WN18RR_", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=1000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=100, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/wn_/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=86835, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("--relation_out_dim", type=int, nargs='+',
                      default=[100, 200], help="relation output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")

    # ## GPU-set
    # args.add_argument("-gpu","--gpu", type=int, default=[0], help="GPU use")
    args = args.parse_args()
    return args

