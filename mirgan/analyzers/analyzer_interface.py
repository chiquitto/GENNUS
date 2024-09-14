class AnalyzerInterface(object):
  def __init__(self):
    """
    Initializes the Analyzer.

    This constructor sets up any necessary parameters or state for the Analyzer.
    """

    pass

  def setInput(self, sequence_list):
    """
    Sets the input data (synthetic samples) to be analyzed.

    This method accepts a list of dicts of sequences that will later be scored. 
    It does not perform any score calculation itself.

    Dict format: {"id": string, "seq": string}

    Args:
        sequence_list (list): A list of sequences to be scored.
    """

    pass

  def prepare(self):
    """
    Prepares the analyzer for scoring.

    This method is called after setInput() and before run(). 
    It is intended to perform any necessary pre-processing before 
    the actual scoring starts.
    """

    pass

  def run(self):
    """
    Calculates the scores for the input sequences.

    This method performs the actual computation of scores for each 
    sequence provided via setInput().
    """

    pass

  def getScores(self):
    """
    Retrieves the calculated scores for each sequence.

    After running the analysis, this method returns a dictionary where 
    the keys are the sequence identifiers and the values are the calculated 
    scores.

    Returns:
        dict: A dictionary with sequence IDs as keys and their corresponding 
        scores as float values, e.g., {"seq1_id": {score: float}, "seq2_id": {score: float}, ...}.
    """
    
    pass

  ###

  @staticmethod
  def list_to_inputdata(listseqs):
    return [ {'id': 'seq_' + str(n), 'seq': seq.upper()} for n, seq in enumerate(listseqs) ]
