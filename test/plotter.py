import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))

from utility import plot_cora_datasplit_graphs, plot_accuracies, plot_final_results

plot_cora_datasplit_graphs()
plot_accuracies('../output/gcnv1-bias-val-accuracies.csv', '../output/plots', 'gcnv1-bias-val-accuracies.png', 'GCN V1 (Bias) Training')
plot_accuracies('../output/gcnv1-no-bias-val-accuracies.csv', '../output/plots', 'gcnv1-no-bias-val-accuracies.png', 'GCN V1 (No Bias) Training')
plot_accuracies('../output/gcnv2-bias-val-accuracies.csv', '../output/plots', 'gcnv2-bias-val-accuracies.png', 'GCN V2 (Bias) Training')
plot_accuracies('../output/gcnv2-no-bias-val-accuracies.csv', '../output/plots', 'gcnv2-no-bias-val-accuracies.png', 'GCN V2 (No Bias) Training')
plot_accuracies('../output/gsv1-mean-val-accuracies.csv', '../output/plots', 'gsv1-mean-val-accuracies.png', 'GraphSAGE V1 (Mean) Training')
plot_accuracies('../output/gsv1-sum-val-accuracies.csv', '../output/plots', 'gsv1-sum-val-accuracies.png', 'GraphSAGE V1 (Sum) Training')
plot_accuracies('../output/gsv1-mean-val-accuracies.csv', '../output/plots', 'gsv1-max-val-accuracies.png', 'GraphSAGE V1 (Max) Training')
plot_accuracies('../output/gsv3-mean-val-accuracies.csv', '../output/plots', 'gsv3-mean-val-accuracies.png', 'GraphSAGE V3 (Mean) Training')
plot_accuracies('../output/gsv3-sum-val-accuracies.csv', '../output/plots', 'gsv3-sum-val-accuracies.png', 'GraphSAGE V3 (Sum) Training')
plot_accuracies('../output/gsv4-mean-val-accuracies.csv', '../output/plots', 'gsv3-max-val-accuracies.png', 'GraphSAGE V3 (Max) Training')
plot_accuracies('../output/gsv4-mean-val-accuracies.csv', '../output/plots', 'gsv4-mean-val-accuracies.png', 'GraphSAGE V4 (Mean) Training')
plot_accuracies('../output/gsv4-sum-val-accuracies.csv', '../output/plots', 'gsv4-sum-val-accuracies.png', 'GraphSAGE V4 (Sum) Training')
plot_accuracies('../output/gsv4-max-val-accuracies.csv', '../output/plots', 'gsv4-max-val-accuracies.png', 'GraphSAGE V4 (Max) Training')
plot_final_results()
