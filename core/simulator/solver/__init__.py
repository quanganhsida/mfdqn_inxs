from . import q_heuristic
from . import optimal
from . import greedy
from . import random
from . import mfdqn
from . import maql
from . import idqn

def create_solver(args):
    if args.solver == 'random':
        return random.Solver(args)
    elif args.solver == 'greedy':
        return greedy.Solver(args)
    elif args.solver == 'optimal':
        return optimal.Solver(args)
    elif args.solver == 'q_heuristic':
        return q_heuristic.Solver(args)
    elif args.solver == 'maql':
        return maql.Solver(args)
    elif args.solver == 'idqn':
        return idqn.Solver(args)
    elif args.solver == 'mfdqn':
        return mfdqn.Solver(args)
    else:
        raise NotImplementedError
