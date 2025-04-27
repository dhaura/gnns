import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../graphSAGE'))

import torch
from utility import  load_reddit_data, get_data_split_masks, train_model, write_accuracies_to_csv
from gsnnv3 import GraphSageNNV3
from gsnnv4 import GraphSageNNV4

output_dir = '../output'
output_file_path = output_dir + '/final-results.csv'

num_iterations = 3

'''
    #### GraphSAGE Version 3 #####
'''

# Load the dataset.
features, num_classes, labels, edge_index = load_reddit_data(add_self_loops=False, adj_type='EI')
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0])

'''
  Train the GraphSage V3 model on the reddit dataset using mean pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV3(input_dim=features.shape[1], hidden_dim=128, output_dim=num_classes, pool_method='mean', droupout=0.5)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GS V3 (Mean),reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gsv3-mean-val-accuracies.csv')

'''
  Train the GraphSage V3 model on the reddit dataset using sum pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV3(input_dim=features.shape[1], hidden_dim=128, output_dim=num_classes, pool_method='sum', droupout=0.5)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GS V3 (Sum),reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gsv3-sum-val-accuracies.csv')

'''
  Train the GraphSage V3 model on the reddit dataset using max pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV3(input_dim=features.shape[1], hidden_dim=128, output_dim=num_classes, pool_method='max', droupout=0.5)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GS V3 (Max),reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gsv3-max-val-accuracies.csv')

'''
    #### GraphSAGE Version 4 #####
'''

# Load the dataset.
features, num_classes, labels, edge_index = load_reddit_data(add_self_loops=True, adj_type='EI')
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0])

'''
  Train the GraphSage V4 model on the reddit dataset using mean pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV4(input_dim=features.shape[1], hidden_dim=128, output_dim=num_classes, pool_method='mean', droupout=0.5)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GS V4 (Mean),reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gsv4-mean-val-accuracies.csv')

'''
  Train the GraphSage V4 model on the reddit dataset using sum pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV4(input_dim=features.shape[1], hidden_dim=128, output_dim=num_classes, pool_method='sum', droupout=0.5)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GS V4 (Sum),reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gsv4-sum-val-accuracies.csv')

'''
  Train the GraphSage V4 model on the reddit dataset using max pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV4(input_dim=features.shape[1], hidden_dim=128, output_dim=num_classes, pool_method='max', droupout=0.5)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GS V4 (Max),reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gsv4-max-val-accuracies.csv')
