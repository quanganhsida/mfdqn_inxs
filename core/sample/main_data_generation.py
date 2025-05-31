import numpy as np
import subnetwork_generate


class init_parameters:
    def __init__(self,rng):
        self.num_of_subnetworks = 20
        self.n_subchannel = 4
        self.deploy_length = 20                                 # the length and breadth of the factory area (m)
        self.subnet_radius = 1                                  # the radius of the subnetwork cell (m)
        self.minD = 0.8                                         # minimum distance from device to controller(access point) (m)
        self.minDistance = 2 * self.subnet_radius               # minimum controller to controller distance (m)
        self.rng_value = np.random.RandomState(rng)
        self.bandwidth = 100e6                                  # bandwidth (Hz)
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
        self.mapXPoints = np.linspace(0, self.deploy_length, num=401, endpoint=True)
        self.mapYPoints = np.linspace(0, self.deploy_length, num=401, endpoint=True)
        self.correlationDistance = 5

snapshots = 30
config = init_parameters(0)

print('#### Generating subnetwork ####')
ch_coef, Location = subnetwork_generate.generate_samples(config, snapshots)

CSI = np.save('Channel_matrix_gain', ch_coef)
Loc = np.save('Location_mat', Location)

# ch_coef = np.load('Channel_matrix_gain.npy')
# Location = np.load('Location_mat.npy')
