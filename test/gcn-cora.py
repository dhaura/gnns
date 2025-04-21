import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../gcn'))

from utility import load_data, convert_adj_to_sparse_tensor, get_original_cora_data_split_masks, get_data_split_masks, train_model
from gcnv1 import GCNV1
from gcnv2 import GCNV2
from gcnv3 import GCNV3

cora_datasplit_output_file_path = '../output/gcn-cora-datasplit-output.csv'
# with open(cora_datasplit_output_file_path, 'w') as f:
#     f.write('model,data_split,accuracy,elapsed_time,num_epochs_to_converge\n')

output_file_path = '../output/cora-output.csv'
# with open(output_file_path, 'w') as f:
#     f.write('model,dataset,accuracy,elapsed_time,num_epochs_to_converge\n')

num_iterations = 10

'''
    #### GCN Version 1 #####
'''

# Load the dataset.
features, num_classes, labels, adj_matrix = load_data('../data/cora/cora.cites', '../data/cora/cora.content', add_self_loops=True)

'''
  Train the GCN V1 model with bias on the cora dataset using the original data split.
'''
# Get split masks.
train_mask, test_mask, val_mask = get_original_cora_data_split_masks(features.shape[0])

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV1(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=True)
    accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(cora_datasplit_output_file_path, 'a') as f:
    f.write(f'gcnv1-bias,original,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GCN V1 model without bias on the cora dataset using the original data split.
'''
# Get split masks.
train_mask, test_mask, val_mask = get_original_cora_data_split_masks(features.shape[0])

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV1(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=False)
    accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(cora_datasplit_output_file_path, 'a') as f:
    f.write(f'gcnv1-no-bias,original,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GCN V1 model with bias on the cora dataset using a random data split.
'''
# Get split masks.
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0], train_ratio=0.6, test_ratio=0.2, shuffle=True)

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV1(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=True)
    accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(cora_datasplit_output_file_path, 'a') as f:
    f.write(f'gcnv1-bias,custom,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

with open(output_file_path, 'a') as f:
    f.write(f'gcnv1-bias,cora,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GCN V1 model without bias on the cora dataset using a random data split.
'''
# Get split masks.
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0], train_ratio=0.6, test_ratio=0.2, shuffle=True)

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV1(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=False)
    accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(cora_datasplit_output_file_path, 'a') as f:
    f.write(f'gcnv1-no-bias,custom,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

with open(output_file_path, 'a') as f:
    f.write(f'gcnv1-no-bias,cora,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')


'''
    #### GCN Version 2 #####
'''

# Load the dataset.
features, num_classes, labels, adj_matrix = load_data('../data/cora/cora.cites', '../data/cora/cora.content', add_self_loops=False)

'''
  Train the GCN V2 model with bias on the cora dataset using the original data split.
'''
# Get split masks.
train_mask, test_mask, val_mask = get_original_cora_data_split_masks(features.shape[0])

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV2(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=True)
    accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(cora_datasplit_output_file_path, 'a') as f:
    f.write(f'gcnv2-bias,original,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GCN V2 model without bias on the cora dataset using the original data split.
'''
# Get split masks.
train_mask, test_mask, val_mask = get_original_cora_data_split_masks(features.shape[0])

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV2(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=False)
    accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(cora_datasplit_output_file_path, 'a') as f:
    f.write(f'gcnv2-no-bias,original,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GCN V2 model with bias on the cora dataset using a random data split.
'''
# Get split masks.
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0], train_ratio=0.6, test_ratio=0.2, shuffle=True)

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV2(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=True)
    accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(cora_datasplit_output_file_path, 'a') as f:
    f.write(f'gcnv2-bias,custom,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

with open(output_file_path, 'a') as f:
    f.write(f'gcnv2-bias,cora,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GCN V2 model without bias on the cora dataset using a random data split.
'''
# Get split masks.
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0], train_ratio=0.6, test_ratio=0.2, shuffle=True)

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(num_iterations):
    # Initialize the GCN model.
    gcn = GCNV2(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=False)
    accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / num_iterations
average_elapsed_time = elapsed_time_sum / num_iterations
average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations

with open(cora_datasplit_output_file_path, 'a') as f:
    f.write(f'gcnv2-no-bias,custom,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

with open(output_file_path, 'a') as f:
    f.write(f'gcnv2-no-bias,cora,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')


# '''
#     #### GCN Version 3 #####
# '''

# # Load the dataset.
# features, num_classes, labels, adj_matrix = load_data('../data/cora/cora.cites', '../data/cora/cora.content', add_self_loops=True)
# adj_matrix = convert_adj_to_sparse_tensor(adj_matrix)

# '''
#   Train the GCN V3 model with bias on the cora dataset using the original data split.
# '''
# # Get split masks.
# train_mask, test_mask, val_mask = get_original_cora_data_split_masks(features.shape[0])

# accuracy_sum = 0
# elapsed_time_sum = 0
# num_epochs_to_converge_sum = 0

# for i in range(num_iterations):
#     # Initialize the GCN model.
#     gcn = GCNV3(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, dropout=0.5, use_bias=True)
#     accuracy, elapsed_time, converged_epoch = train_model(gcn, features, adj_matrix, labels, train_mask, val_mask, test_mask, num_epochs=200)
#     accuracy_sum += accuracy
#     elapsed_time_sum += elapsed_time
#     num_epochs_to_converge_sum += converged_epoch

# # Calculate the average accuracy, elapsed time, and number of epochs to converge.
# average_accuracy = accuracy_sum / num_iterations
# average_elapsed_time = elapsed_time_sum / num_iterations
# average_num_epochs_to_converge = num_epochs_to_converge_sum / num_iterations
# print(f'average_accuracy: {average_accuracy}, average_elapsed_time: {average_elapsed_time}, average_num_epochs_to_converge: {average_num_epochs_to_converge}')

# with open(cora_datasplit_output_file_path, 'a') as f:
#     f.write(f'gcnv3-bias,original,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')
