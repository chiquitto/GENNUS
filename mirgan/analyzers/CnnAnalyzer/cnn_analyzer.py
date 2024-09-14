import numpy as np
import sys
import torch

# setting path
sys.path.append('../')

from ..analyzer_interface import AnalyzerInterface
from .cnnanalyzer_utils import contains_certain_characters, generate_onehot_encoder, \
    tokenizer, onehot_tokenized
from analyzers.CnnAnalyzer.structure.BinaryClassificator import BinaryClassificator

class CnnAnalyzer(AnalyzerInterface):
    def __init__(self, model_path,
        device, maxlen=121, padnt="N"):

        self.maxlen = maxlen
        self.padnt = padnt
        self.charmap = {'N':0, 'A':1, 'T':2, 'G':3, 'C':4}
        self.device = device

        self.init_classificator(model_path)

    def setInput(self, sequence_list):
        self.sequence_list = sequence_list

    def prepare(self):
        self.reset_scores()
        self.sequence_list_prepared = self.filter_sequences(self.sequence_list)

    def run(self):
        onehot = generate_onehot_encoder(len(self.charmap))

        tokenized_seqs = [ tokenizer(item['seq'], self.charmap) for item in self.sequence_list_prepared ]
        prepared_data = onehot_tokenized(np.array(tokenized_seqs, dtype=np.float32), onehot)

        self.run_classificator(prepared_data)

    def getScores(self):
        return self.scores
    
    ###

    def run_classificator(self, data):
        y_pred = self.run_classificator_(data)

        for item, score in zip(self.sequence_list_prepared, y_pred):
            self.scores[item['id']]['score'] = float(score)

        self.mean_score = float(y_pred.mean())
    
    def run_classificator_(self, x_values):
        x_values = torch.Tensor(x_values).to(device=self.device)

        y_pred = self.classificator(x_values)
        y_pred = y_pred.reshape(-1)
        y_pred = y_pred.detach().cpu().numpy()

        return y_pred
    
    def run_first(self, initial_data):
        initial_data = self.list_to_inputdata(initial_data)
        self.setInput(initial_data)
        self.prepare()
        self.run()

    def init_classificator(self, model_path):
        self.classificator = BinaryClassificator( len(self.charmap), self.maxlen, 512 ).to(device=self.device)
        self.classificator.load_state_dict(torch.load(model_path))
        self.classificator.eval()

    def reset_scores(self):
        self.scores = {}
        self.mean_score = {}

        for row in self.sequence_list:
            self.scores[row['id']] = {'score': 0}

    def filter_sequences(self, seqs):
        filtered_seqs = []
        allowed = ('A', 'T', 'G', 'C')
        for item in seqs:
            seq = item['seq'].upper().replace('U', 'T')
            
            if not contains_certain_characters(seq, allowed):
                continue

            seq = seq.ljust(self.maxlen, self.padnt)[:self.maxlen]
            filtered_seqs.append({ 'id': item['id'], 'seq': seq })
        return filtered_seqs