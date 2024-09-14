from mirgan_utils import gpu_test
use_cuda, device = gpu_test()

# path to your downloaded/trained CNN Analyzer model
CNNANALYZER_MODEL_PATH='analyzers/CnnAnalyzer/cnnanalyzer_models/cnnanalyzer.pt'

from analyzers.CnnAnalyzer.cnn_analyzer import CnnAnalyzer
analyzer = CnnAnalyzer(
    model_path=CNNANALYZER_MODEL_PATH,
    device=device,
    maxlen=121,
    padnt='N'
)
analyzer_threshold = 0.95

if __name__ == "__main__":
    # Example: How to use CNN Analyzer

    with open('input/example_gan_input.txt') as input_file:
        lines = [{'id':'seq%04d'%k, 'seq':next(input_file).strip()} for k in range(10)]
    
    analyzer.setInput(lines)
    analyzer.prepare()
    analyzer.run()
    print(analyzer.getScores())
