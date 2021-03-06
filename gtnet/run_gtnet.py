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
    'predict': Command('run.predict',
                       'Predict taxonomy of provided sequence(s)'),
    'run-test': Command('run.run_test', 'Run gtnet on a sample dataset provided'),
}


def print_help():
    print('Usage: gtnet <command> [options]')
    print('Available commands are:\n')
    for c, f in command_dict.items():
        nspaces = 16 - len(c)
        print(f'    {c}' + ' '*nspaces + f.doc)
    print('    help            print this usage statememt')
    print()


def gtnet_run():
    if len(sys.argv) == 1 or sys.argv[1] in ('help', '-h', '--help'):
        print_help()
    else:
        cmd = sys.argv[1]
        func = command_dict[cmd].get_func()
        func(sys.argv[2:])
