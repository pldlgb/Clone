import torch
from model.models import SpKBGATConvOnly,SpKBGATModified

def evaluate_conv(args, Corpus, unique_entities, entity_embeddings, relation_embeddings):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_conv, args.alpha_conv, args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)), strict=False)

    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        Corpus.get_validation_pred(args, model_conv, unique_entities)


### It doesn't work 
def evaluate_gat(args, Corpus, unique_entities, entity_embeddings, relation_embeddings):
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)
    model_gat.load_state_dict(torch.load(
        '{}/trained_{}.pth'.format(args.output_folder, args.epochs_gat - 1)), strict=False)

    model_gat.cuda()
    model_gat.eval()
    with torch.no_grad():
        Corpus.get_validation_pred(args, model_gat, unique_entities)