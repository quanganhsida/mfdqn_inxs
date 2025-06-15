import numpy as np
import os

from . import util

def prepare_data(args):

    class init_parameters:
        def __init__(self, rng):
            self.num_of_subnetworks = args.n_subnetwork
            self.n_subchannel = args.n_channel
            self.deploy_length = args.deploy_length                 # the length and breadth of the factory area (m)
            self.subnet_radius = args.subnetwork_radius             # the radius of the subnetwork cell (m)
            self.minD = 0.8                                         # minimum distance from device to controller(access point) (m)
            self.minDistance = 2 * self.subnet_radius               # minimum controller to controller distance (m)
            self.rng_value = np.random.RandomState(rng)
            self.bandwidth = args.bandwidth
            self.ch_bandwidth = self.bandwidth / self.n_subchannel
            self.fc = 6e9                                          # Carrier frequency (Hz)
            self.lambdA = 3e8/self.fc
            self.clutType = 'dense'                                 # Type of clutter (sparse or dense)
            self.clutSize = 2.0                                     # Clutter element size [m]
            self.clutDens = 0.6                                     # Clutter density [%]
            self.shadStd = 7.2                                      # Shadowing std (NLoS)
            self.max_power = 1
            self.no_dbm = -174
            self.noise_figure_db = 5
            self.noise_power = 10 ** ((self.no_dbm + self.noise_figure_db + 10 * np.log10(self.ch_bandwidth)) / 10)
            self.mapXPoints = np.linspace(0, self.deploy_length, num=20 * self.deploy_length, endpoint=True)
            self.mapYPoints = np.linspace(0, self.deploy_length, num=20 * self.deploy_length, endpoint=True)
            self.correlationDistance = 5

    snapshots = args.n_snapshot
    config = init_parameters(args.seed)

    print(f'[+] generating ap {args.n_subnetwork=} {args.n_channel=} {args.n_snapshot=}')
    label = f'{args.n_subnetwork}_{args.deploy_length}'
    csi, ap_location, client_location = util.generate_samples(config, snapshots)
    path = os.path.join(args.data_dir, f'csi_{label}.npy')
    np.save(path, csi)
    path = os.path.join(args.data_dir, f'ap_location_{label}.npy')
    np.save(path, ap_location)
    path = os.path.join(args.data_dir, f'client_location_{label}.npy')
    np.save(path, client_location)
#     angle = np.random.uniform(0, 2 * np.pi, args.n_snapshot * args.n_subnetwork)
#     radius = np.random.uniform(0.8, 1, args.n_snapshot * args.n_subnetwork)
#     x = (radius * np.cos(angle)).reshape(args.n_snapshot, args.n_subnetwork)
#     y = (radius * np.sin(angle)).reshape(args.n_snapshot, args.n_subnetwork)
#     client_location = np.zeros([args.n_snapshot, args.n_subnetwork, 2])
#     client_location[:, :, 0] = ap_location[:, :, 0] + x
#     client_location[:, :, 1] = ap_location[:, :, 1] + y
