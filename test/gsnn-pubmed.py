import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../graphSAGE'))

import time
import torch
import torch.nn.functional as F
from utility import convert_pubmed_to_txt, load_data, get_accuracy, get_data_split_masks, transfer_data_to_device, train_model
from gsnnv1 import GraphSageNNV1
from gsnnv2 import GraphSageNNV2
from gsnnv3 import GraphSageNNV3
from gsnnv4 import GraphSageNNV4

output_file_path = '../output/pubmed-output.csv'


'''
    #### GraphSAGE Version 1 #####
'''

# Load the dataset.
features, num_classes, labels, adj_list = load_data('../data/pubmed/pubmed.cites', '../data/pubmed/pubmed.content', sep=' ', add_self_loops=False, adj_type='AL')
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0])

'''
  Train the GraphSage V1 model on the pubmed dataset using mean pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV1(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_func=torch.mean, droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, adj_list, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv1-mean,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GraphSage V1 model on the pubmed dataset using sum pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV1(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_func=torch.sum, droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, adj_list, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv1-sum,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GraphSage V1 model on the pubmed dataset using max pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV1(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_func=lambda x, dim: torch.max(x, dim=dim).values, droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, adj_list, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv1-max,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')


'''
    #### GraphSAGE Version 2 #####
'''

# Load the dataset.
features, num_classes, labels, adj_list = load_data('../data/pubmed/pubmed.cites', '../data/pubmed/pubmed.content', sep=' ', add_self_loops=True, adj_type='AL')
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0])

'''
  Train the GraphSage V2 model on the pubmed dataset using mean pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV2(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_func=torch.mean, droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, adj_list, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv2-mean,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GraphSage V2 model on the pubmed dataset using sum pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV2(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_func=torch.sum, droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, adj_list, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv12sum,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GraphSage V2 model on the pubmed dataset using max pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV2(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_func=lambda x, dim: torch.max(x, dim=dim).values, droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, adj_list, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv2-max,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')


'''
    #### GraphSAGE Version 3 #####
'''

# Load the dataset.
features, num_classes, labels, edge_index = load_data('../data/pubmed/pubmed.cites', '../data/pubmed/pubmed.content', sep=' ', add_self_loops=False, adj_type='EI')
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0])

'''
  Train the GraphSage V3 model on the pubmed dataset using mean pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV3(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_method='mean', droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv3-mean,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GraphSage V3 model on the pubmed dataset using sum pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV3(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_method='sum', droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv3-sum,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GraphSage V3 model on the pubmed dataset using max pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV3(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_method='max', droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv3-max,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')


'''
    #### GraphSAGE Version 4 #####
'''

# Load the dataset.
features, num_classes, labels, edge_index = load_data('../data/pubmed/pubmed.cites', '../data/pubmed/pubmed.content', sep=' ', add_self_loops=True, adj_type='EI')
train_mask, test_mask, val_mask = get_data_split_masks(features.shape[0])

'''
  Train the GraphSage V4 model on the pubmed dataset using mean pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV4(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_method='mean', droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv4-mean,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GraphSage V4 model on the pubmed dataset using sum pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV4(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_method='sum', droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv4-sum,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')

'''
  Train the GraphSage V4 model on the pubmed dataset using max pooling.
'''

accuracy_sum = 0
elapsed_time_sum = 0
num_epochs_to_converge_sum = 0

for i in range(10):
    # Initialize the Graph SAGE model.
    gsnn = GraphSageNNV4(input_dim=features.shape[1], hidden_dim=16, output_dim=num_classes, pool_method='max', droupout=0.5)
    accuracy, elapsed_time, converged_epoch = train_model(gsnn, features, edge_index, labels, train_mask, val_mask, test_mask, num_epochs=200)
    accuracy_sum += accuracy
    elapsed_time_sum += elapsed_time
    num_epochs_to_converge_sum += converged_epoch

# Calculate the average accuracy, elapsed time, and number of epochs to converge.
average_accuracy = accuracy_sum / 10
average_elapsed_time = elapsed_time_sum / 10
average_num_epochs_to_converge = num_epochs_to_converge_sum / 10

with open(output_file_path, 'a') as f:
    f.write(f'graphSAGEv4-max,{average_accuracy},{average_elapsed_time},{average_num_epochs_to_converge}\n')
