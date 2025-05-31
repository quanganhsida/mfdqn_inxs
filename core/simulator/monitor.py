import pandas as pd
import numpy as np
import torch
import tqdm
import os

class Monitor:

    def __init__(self, args, progress_bar=True):
        # save
        self.args = args
        # initialize progress bar
        if progress_bar:
            if args.mode == 'train':
                T = args.n_train_step
            elif args.mode == 'test':
                T = args.n_test_step
            if progress_bar:
                self.bar = tqdm.tqdm(range(T))
            # initialize writer
            self.csv_data = {}
            self.global_step = 0

    def __update_time(self):
        self.bar.update(1)

    def __display(self):
        self.bar.display()

    def step(self, info):
        # update progress bar
        self.__update_time()
        self.__update_description(**info)
        self.__display()
        # log to csv
        self.__update_csv(info)
        self.global_step += 1

    def __update_csv(self, info):
        for key in info.keys():
            if key not in self.csv_data:
                self.csv_data[key] = [float(info[key])]
            else:
                self.csv_data[key].append(float(info[key]))

    def export_csv(self):
        # extract args
        args = self.args
        directory = os.path.join(args.csv_dir, args.mode)
        os.makedirs(directory, exist_ok=True)
        # save data to csv
        path = os.path.join(args.csv_dir, args.mode, f'{self.label}.csv')
        df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=None)

    ####################################################################################
    # MODIFY HERE
    ####################################################################################
    @property
    def label(self):
        args = self.args
        label = f'{args.solver}'
        return label

    def __update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            for term in ['reward', 'loss', 'eps']:
                if term in key:
                    _kwargs[key] = f'{kwargs[key]:0.3f}'
        self.bar.set_postfix(**_kwargs)
    ####################################################################################
    # END
    ####################################################################################
