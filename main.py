import os
os.environ["CUDA_VISIBLE_DEVICES"]='5'
import pickle
from copy import deepcopy

from utils.arg_set import parse_args
from process.load_data import load_data
from train.gnn import train_gat
from train.conv import train_conv
from train.fusion import fusionKGC

from eval.eval import evaluate_conv

if __name__ == '__main__':
    args= parse_args()
    Corpus, entity_embeddings, relation_embeddings = load_data(args)
    print(args.data)
    ## use n-hop information
    if(args.get_2hop):
        file = args.data + "/2hop.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(Corpus.node_neighbors_2hop, handle,
                        protocol = pickle.HIGHEST_PROTOCOL)
    if(args.use_2hop):
        print("Opening node_neighbors pickle object")
        file = args.data + "/2hop.pickle"
        with open(file, 'rb') as handle:
            node_neighbors_2hop = pickle.load(handle)

    entity_embeddings_copied = deepcopy(entity_embeddings)
    relation_embeddings_copied = deepcopy(relation_embeddings)

    print("Initial entity dimensions {} , relation dimensions {}".format(
        entity_embeddings.size(), relation_embeddings.size()))

    # train_gat(args, Corpus, entity_embeddings, relation_embeddings)
    train_conv(args, Corpus, entity_embeddings, relation_embeddings)
    # fusionKGC(args, Corpus, entity_embeddings, relation_embeddings)
    evaluate_conv(args, Corpus, Corpus.unique_entities_train, entity_embeddings, relation_embeddings)