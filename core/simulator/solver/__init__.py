from . import q_heuristic
from . import optimal
from . import greedy
from . import random
from . import mfdqn
from . import maql
from . import idqn
from . import vdn

def create_solver(args):
    return eval(args.solver).Solver(args)
