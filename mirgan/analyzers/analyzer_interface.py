class AnalyzerInterface(object):
  def __init__(self):
    # print('AnalyzerInterface')
    pass

  def setInput(self, sequence_list):
    pass

  def prepare(self):
    pass

  def run(self):
    pass

  def getScores(self):
    # return should be:
    # dict {"seq1_id": float, "seq2_id": float, ...}
    pass

  ###

  @staticmethod
  def list_to_inputdata(listseqs):
    return [ {'id': 'seq_' + str(n), 'seq': seq.upper()} for n, seq in enumerate(listseqs) ]
