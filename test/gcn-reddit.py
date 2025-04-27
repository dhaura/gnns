import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../gcn'))

from utility import load_reddit_data, get_data_split_masks, train_model,write_accuracies_to_csv
from gcnv1 import GCNV1
from gcnv2 import GCNV2

output_dir = '../output'
output_file_path = output_dir + '/final-results.csv'

num_iterations = 3

'''
    #### GCN Version 1 #####
'''

# Load the dataset.
features, num_classes, labels, adj_matrix = load_reddit_data(add_self_loops=True, adj_type='AM')
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0], train_ratio=0.6, test_ratio=0.2, shuffle=True)

'''
  Train the GCN V1 model with bias on the reddit dataset.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV1(input_dim=features.shape[1], hidden_dim=128, output_dim=num_classes, dropout=0.2, use_bias=True)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GCN V1 (B),reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gcnv1-bias-val-accuracies.csv')

'''
  Train the GCN V1 model without bias on the reddit dataset.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV1(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=False)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GCN V1,reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gcnv1-no-bias-val-accuracies.csv')


'''
    #### GCN Version 2 #####
'''

# Load the dataset.
features, num_classes, labels, adj_matrix = load_reddit_data(add_self_loops=False, adj_type='AM')
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0], train_ratio=0.6, test_ratio=0.2, shuffle=True)

'''
  Train the GCN V2 model with bias on the reddit dataset.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV2(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=True)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GCN V2 (B),reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gcnv2-bias-val-accuracies.csv')

'''
  Train the GCN V2 model without bias on the cora dataset.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV2(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=False)
    val_accuracies, accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(output_file_path, 'a') as f:
    f.write(f'GCN V2,reddit,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

write_accuracies_to_csv('reddit', val_accuracies, output_dir + '/gcnv2-no-bias-val-accuracies.csv')
