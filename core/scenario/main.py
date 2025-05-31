import simulator

def main(args):
    solver = simulator.create_solver(args)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:
        raise NotImplementedError
