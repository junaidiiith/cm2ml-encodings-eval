import os
import torch

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



BERT_MODEL = 'bert-base-uncased'
MODERN_BERT = 'answerdotai/ModernBERT-base'
WORD2VEC_MODEL = 'word2vec'
TFIDF_MODEL = 'tfidf'
FAST_TEXT_MODEL = 'uml-fasttext.bin'

W2V_CONFIG = dict(
    epoch=100,
    dim=128,
    ws=5,
    minCount=1,
    thread=4,
    model='skipgram'
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_float32_matmul_precision('high')


seed = 42
datasets_dir = 'datasets'
ecore_json_path = os.path.join(datasets_dir, 'ecore_555/ecore_555.jsonl')
mar_json_path = os.path.join(datasets_dir, 'mar-ecore-github/ecore-github.jsonl')
modelsets_uml_json_path = os.path.join(datasets_dir, 'modelset/uml.jsonl')
modelsets_ecore_json_path = os.path.join(datasets_dir, 'modelset/ecore.jsonl')


graph_data_dir = 'datasets/graph_data'
results_dir = 'results'

# Path: settings.py


EDGE_CLS_TASK = 'edge_cls'
LINK_PRED_TASK = 'lp'
NODE_CLS_TASK = 'node_cls'
GRAPH_CLS_TASK = 'graph_cls'
DUMMY_GRAPH_CLS_TASK = 'dummy_graph_cls'


SEP = ' '
REFERENCE = 'reference'
SUPERTYPE = 'supertype'
CONTAINMENT = 'containment'


EPOCH = 'epoch'
LOSS = 'loss'
TRAIN_LOSS = 'train_loss'
TEST_LOSS = 'test_loss'
TEST_ACC = 'test_acc'

TRAINING_PHASE = 'train'
VALIDATION_PHASE = 'val'
TESTING_PHASE = 'test'


