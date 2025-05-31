import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_network(args):
    # load ap csi
    path = os.path.join(args.data_dir, 'csi.npy')
    ap_csi = np.load(path) # n_snapshot, n_subchannel, n_subnetwork, n_subnetwork
    # load ap location
    path = os.path.join(args.data_dir, 'ap_location.npy')
    ap_location = np.load(path) # n_snapshot, n_subnetwork, 2d
    # load client location
    path = os.path.join(args.data_dir, 'client_location.npy')
    client_location = np.load(path) # n_snapshot, n_subnetwork, 2d
    # plot subnetwork coverage
    fig, ax = plt.subplots(figsize=(10, 10))
    t = 0
    for n in range(args.n_subnetwork):
        x, y = ap_location[t, n, :]
        print(x, y)
        circle = patches.Circle((x, y), args.subnetwork_radius,
                                edgecolor='blue', facecolor='lightblue')
        ax.add_patch(circle)
    plt.scatter(ap_location[t, :, 0], ap_location[t, :, 1], c='b', s=5)
    plt.scatter(client_location[t, :, 0], client_location[t, :, 1], c='r', s=1)
    # save figure
    plt.xlim((0, 20))
    plt.ylim((0, 20))
    # decorate
    plt.tight_layout()
    path = os.path.join(args.figure_dir, f'{args.scenario}.pdf')
    plt.savefig(path)
