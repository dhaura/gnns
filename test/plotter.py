import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))

from utility import plot_cora_datasplit_graphs, plot_metrics_from_csv

plot_cora_datasplit_graphs()
plot_metrics_from_csv('../output/concat-output.csv', '../output/plots/')
