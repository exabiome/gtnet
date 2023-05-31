import io
import sys
from importlib import import_module


class Command:
    def __init__(self, module, doc):
        ar = f'gtnet.{module}'.split('.')
        self.pkg = '.'.join(ar[:-1])
        self.func = ar[-1]
        self.doc = doc

    def get_func(self):
        return getattr(import_module(self.pkg), self.func)


command_dict = {
    "classify": Command("classify.classify",
                        "Perform taxonomic classificaiton for all sequences"),
    "predict":  Command("predict.predict",
                        "Predict taxonomy of all sequences at each taxonomic level"),
    "filter":   Command("filter.filter",
                        "Filter taxonomic classifications from the predict command"),
    "test":     Command("run.run_test", "Run gtnet on a sample dataset provided"),
}


def print_help():
    sio = io.StringIO()
    print('Usage: gtnet <command> [options]', file=sio)
    print('Available commands are:\n', file=sio)
    for c, f in command_dict.items():
        nspaces = 16 - len(c)
        print(f'    {c}' + ' '*nspaces + f.doc, file=sio)
    print('    help            print this usage statememt\n', file=sio)
    sys.stdout.write(sio.getvalue())


def run():
    if len(sys.argv) == 1 or sys.argv[1] in ('help', '-h', '--help'):
        print_help()
    else:
        cmd = sys.argv[1]
        func = command_dict[cmd].get_func()
        func(sys.argv[2:])

if __name__ == '__main__':
    run()
