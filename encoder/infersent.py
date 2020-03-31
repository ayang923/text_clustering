from .models import InferSent
import torch

def infersent_train(sentences):
    V = 1
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    
    W2V_PATH = 'encoder/GloVe/glove.840B.300d.txt'
    infersent.set_w2v_path(W2V_PATH)
    
    infersent.build_vocab_k_words(K=50000)
    
    return infersent
