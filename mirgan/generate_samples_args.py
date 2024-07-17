import getopt
import sys

def usage():
    # python mirgan/generate_samples.py --generator generator.pt --samples 3000 --output output.txt
    s = "USAGE: " \
        + "python mirgan4.py " \
        + "--generator generator pt " \
        + "--samples 3000 " \
        + "--output output.txt"
    print(s)
    
def process_argv(argv):

    requireds = ["generator", "samples", "output"]
    input_args = requireds + ['help']

    try:
        longopts = [ opt + "=" for opt in input_args ]
        opts, args = getopt.getopt(argv[1:], "", longopts)
    except getopt.GetoptError as e:
        print("Wrong usage!")
        print(e)
        usage()
        sys.exit(1)

    # parse the options
    r = {}
    for op, value in opts:
        op = op.replace('--', '')
        if op == 'help':
            usage()
            sys.exit()
        elif op in input_args:
            r[op] = value

    for required in requireds:
        if not required in r:
            print("Wrong usage!!")
            print("Param {} is required".format(required))
            usage()
            sys.exit(1)

    return r

def is_notebook() -> bool:
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter