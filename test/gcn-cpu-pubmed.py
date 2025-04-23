from isplib import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../gcn'))

from utility import load_data, convert_adj_to_sparse_tensor, get_data_split_masks, train_model, write_accuracies_to_csv
from gcnv1 import GCNV1
from gcnv3 import GCNV3

output_dir = '../output'
gcn_cpu_results_file_path = output_dir + '/gcn-cpu-results.csv'

num_iterations = 10

'''
    #### GCN Version 1 #####
'''

# Load the dataset.
features, num_classes, labels, adj_matrix = load_data('../data/pubmed/pubmed.cites', '../data/pubmed/pubmed.content', sep=' ', add_self_loops=True)
# Get split masks.
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0], train_ratio=0.6, test_ratio=0.2, shuffle=True)

'''
  Train the GCN V1 model on the pubmed dataset.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV1(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=True)
    _, accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, use_cpu=True)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(gcn_cpu_results_file_path, 'a') as f:
    f.write(f'GCN V1,pubmed,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
    #### GCN Version 3 #####
'''

# Construct the adjacency matrix.
adj_matrix = convert_adj_to_sparse_tensor(adj_matrix)

'''
  Train the GCN V3 model on the pubmed dataset without using iSpLib patch.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV3(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=True)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, use_cpu=True)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations
print(f'average_accuracy: {average_accuracy}, average_elapsed_time: {average_elapsed_time}, average_num_epochs_to_converge: {average_num_epochs_to_converge}')

with open(gcn_cpu_results_file_path, 'a') as f:
    f.write(f'GCN V3,pubmed,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GCN V3 model on the pubmed dataset using iSpLib patch.
'''
iSpLibPlugin.patch_pyg()

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV3(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=True)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, use_cpu=True)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations
print(f'average_accuracy: {average_accuracy}, average_elapsed_time: {average_elapsed_time}, average_num_epochs_to_converge: {average_num_epochs_to_converge}')

with open(gcn_cpu_results_file_path, 'a') as f:
    f.write(f'GCN V3 (iSpLib),pubmed,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('pubmed', val_accuracies, output_dir + '/gcnv3-val-accuracies.csv')

iSpLibPlugin.unpatch_pyg()
