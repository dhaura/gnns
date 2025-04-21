# Helper functions to load data and, develop and train the GCN.
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
import os
import time


'''
    Normalize the adjacency matrix using symmetric normalization,
    using D^-0.5 * A * D^-0.5.
'''
def normalize_adj(adj):
    deg = np.array(adj.sum(1)).flatten() # Compute the degree matrix (D).
    deg_inv_sqrt = np.power(deg, -0.5) # Compute D^(-0.5).
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0 # Handle division by zero (if a node has no edges).
    D_inv_sqrt = sp.diags(deg_inv_sqrt) # Convert D^(-0.5) into a diagonal matrix.

    # Compute the normalized adjacency matrix in COO (Coordinate Format), sparse matrix format.
    return (D_inv_sqrt @ adj @ D_inv_sqrt).tocoo()

def convert_adj_to_sparse_tensor(adj):
    # Extract indices and values from adj_tensor
    coo = adj.coalesce()
    row, col = coo.indices()
    value = coo.values()

    # Convert to torch_sparse SparseTensor
    adj = SparseTensor(
        row=row,
        col=col,
        value=value,
        sparse_sizes=adj.size()
    )
    return adj

'''
    Normalize the feature matrix by dividing each feature by the sum of its row.
'''
def normalize_features(features):
    row_sum = features.sum(1, keepdims=True)
    row_sum[row_sum == 0] = 1  # Handle division by zero.
    return features / row_sum

'''
    Convert the edge list into a normalized adjacency matrix.
'''
def get_adjacency_matrix(edge_list, num_nodes, add_self_loops=False):
    # Create an undirected NetworkX graph using the edge list.
    graph = nx.Graph()
    graph.add_edges_from(edge_list.values)

     # Extract the adjacency matrix from the graph.
    adj = nx.adjacency_matrix(graph, nodelist=range(num_nodes))
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0])  # Add self-loops.

    # Normalize adjacency matrix.
    normalized_adj_matrix = normalize_adj(adj)

    # Convert normalized sparse adjacency matrix into a sparse tensor.
    adj_tensor = torch.sparse.FloatTensor(
        torch.LongTensor([normalized_adj_matrix.row, normalized_adj_matrix.col]),
        torch.FloatTensor(normalized_adj_matrix.data),
        torch.Size(normalized_adj_matrix.shape)
    )

    return adj_tensor

def get_adjacency_list(edge_list, num_nodes, add_self_loops=False):
    if add_self_loops:
        adj_list = [[i] for i in range(num_nodes)]
    else:
        adj_list = [[] for _ in range(num_nodes)]

    for u, v in edge_list.values:
        adj_list[u].append(v)
        adj_list[v].append(u)

    return adj_list

def get_symmetric_edge_index(edge_list, add_self_loops=False):
    edge_index = torch.tensor(edge_list.values, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = edge_index.unique(dim=1)

    if add_self_loops:
        self_loops = torch.arange(edge_index.max() + 1).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

    return edge_index

'''
    Load the given dataset and convert it into tensors.
    Output: feature list, label list, normalized adjacency matrix.
'''
def load_data(edge_list_path, node_data_path, sep='\t', add_self_loops=False, adj_type='AM'):
    # Read the edge list and node data (features and labels) of the dataset.
    edge_list = pd.read_csv(edge_list_path, sep=sep, header=None, names=['source', 'target'])
    node_data = pd.read_csv(node_data_path, sep=sep, header=None)

    # Extract the node labels and features from node data.
    node_labels = node_data.iloc[:, -1]
    node_features = node_data.iloc[:, 1:-1]

    # Normalize features and convert them to tensor.
    features = normalize_features(node_features.values)
    features = torch.tensor(features, dtype=torch.float32)

    # Count the number of unique labels present.
    unique_labels = np.unique(node_labels)
    num_of_labels = len(np.unique(node_labels))

    # Map each label to a unique integer (from 0 -> number of unique labels).
    label_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_id_map[label] for label in node_labels]

    # Convert labels to a tensor.
    labels = torch.tensor(labels, dtype=torch.long)

    # Extract node ids from node data and map node ids to a unique integer (from 0 -> number of nodes).
    node_ids = np.array(node_data.iloc[:, 0], dtype=np.int32)
    node_id_map = {j: i for i, j in enumerate(node_ids)}

    # Apply new integer node ids to the edge list.
    edge_list['source'] = edge_list['source'].map(node_id_map)
    edge_list['target'] = edge_list['target'].map(node_id_map)

    if adj_type == 'AM':
        # Get the normalized adjacency matrix as tensor from the edge list.
        adj = get_adjacency_matrix(edge_list, len(node_ids), add_self_loops)
    elif adj_type == 'AL':
        adj = get_adjacency_list(edge_list, len(node_ids), add_self_loops)
    elif adj_type == 'EI':
        adj = get_symmetric_edge_index(edge_list, add_self_loops)
    else:
        raise ValueError("Invalid adjacency type. Choose from 'AM', 'AL', or 'EI'.")

    return features, num_of_labels, labels, adj

'''
    Retrieve the train, test and validation masks for the given dataset.
'''
def get_data_split_masks(num_of_nodes, train_ratio=0.6, test_ratio=0.2, shuffle=True):
    train_num_nodes = int(num_of_nodes * train_ratio)
    test_num_nodes = int(num_of_nodes * test_ratio)
    val_num_nodes = num_of_nodes - train_num_nodes - test_num_nodes

    if shuffle:
        indices = torch.randperm(num_of_nodes)
    else:
        indices = np.arange(num_of_nodes)

    train_mask = torch.zeros(num_of_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_of_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_of_nodes, dtype=torch.bool)

    train_mask[indices[:train_num_nodes]] = True
    test_mask[indices[train_num_nodes:train_num_nodes+test_num_nodes]] = True
    val_mask[indices[train_num_nodes+test_num_nodes:]] = True

    return train_mask, test_mask, val_mask

'''
    Retrieve the train, test and validation masks for the cora dataset
    as it is done in the original GCN.
'''
def get_original_cora_data_split_masks(num_of_nodes):
    train_mask = torch.zeros(num_of_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_of_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_of_nodes, dtype=torch.bool)

    idx_train = torch.arange(140)
    idx_val = torch.arange(200, 500)
    idx_test = torch.arange(500, 1500)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

'''
    Calculate the accuracy of the model for a given output and labels in an epoch.
'''
def get_accuracy(output, labels):
    predictions = output.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total

'''
    Convert Pubmed Diabetes dataset from TAB format to text.
'''
def convert_pubmed_to_txt(
    node_data_file='../data/pubmed/pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab',
    edge_list_file='../data/pubmed/pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab',
    output_dir='../data/pubmed'
):
    os.makedirs(output_dir, exist_ok=True)

    feature_list = []
    feature_index = {}

    # Parse header line to get feature names.
    print("Extracting feature list ...")
    with open(node_data_file, 'r') as nf:
        for line in nf:
            if line.startswith('cat='):
                parts = line.strip().split('\t')
                for token in parts:
                    if token.startswith('numeric:'):
                        fname = token.split(':')[1]
                        feature_list.append(fname)
                break
    
    feature_index = {name: i for i, name in enumerate(feature_list)}
    num_features = len(feature_list)
    print(f"Extracted {num_features} features.")

    id_map = {}
    label_set = set()

    # Parse node features.
    print("\nExtracting node labels and features ...")
    with open(node_data_file, 'r') as nf, open(os.path.join(output_dir, 'pubmed.content'), 'w') as cf:
        for line in nf:
            line = line.strip()
            if not line or line.startswith("NODE") or line.startswith("cat="):
                continue
            if '\t' not in line:
                continue

            paper_id, data = line.split('\t', 1)

            fields = data.split()
            features = ['0.0'] * num_features
            label = None

            for field in fields:
                if '=' not in field:
                    continue
                key, val = field.split('=', 1)
                if key == 'label':
                    label = val
                elif key in feature_index:
                    features[feature_index[key]] = val

            id_map[paper_id] = len(id_map)
            label_set.add(label)

            cf.write(f"{paper_id} {' '.join(features)} {label}\n")

        print("Completed node feature and label extraction.")

    # Parse citation edges and write into 'pubmed.cites'.
    print("\nExtracting edge list ...")
    with open(edge_list_file, 'r') as ef, open(os.path.join(output_dir, 'pubmed.cites'), 'w') as cf:
        for line in ef:
            line = line.strip()
            if not line or line.startswith("DIRECTED") or line.startswith("NO_FEATURES"):
                continue

            parts = line.split('|')
            if len(parts) != 2:
                continue
            left = parts[0].split()[-1].strip()
            right = parts[1].strip()
            src = left.replace("paper:", "")
            dst = right.replace("paper:", "")

            if src in id_map and dst in id_map:
                cf.write(f"{src} {dst}\n")
        print("Completed edge list extraction.")

    print(f"\nConverted to text format at: {output_dir}\n")

'''
    Transfer the data to the specified device (CPU or GPU).
'''
def transfer_data_to_device(device, features, labels, adj, train_mask, test_mask, val_mask):
    features = features.to(device)
    labels = labels.to(device)
    if torch.is_tensor(adj) or isinstance(adj, SparseTensor):
        adj = adj.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    val_mask = val_mask.to(device)

    return features, labels, adj, train_mask, test_mask, val_mask

'''
    Train the GCN model using the given features, adjacency matrix or list/ edge index, labels and masks.
'''
def train_model(gnn, features, adj, labels, train_mask, val_mask, test_mask, num_epochs=200, learning_rate=0.01, weight_decay=5e-4):
    
    # Set torch device type.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    gnn = gnn.to(device)

    # Transfer the data to GPU, if cuda is available.
    features, labels, adj, train_mask, test_mask, val_mask = transfer_data_to_device(device, features, labels, adj, train_mask, test_mask, val_mask)

    optimizer = torch.optim.Adam(gnn.parameters(), lr=learning_rate, weight_decay=weight_decay)

    accuracies = []
    converged = False
    converged_epoch = -1
    start_time = time.time()
    for epoch in range(num_epochs):
        gnn.train()
        optimizer.zero_grad()
        out = gnn(features, adj)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        train_accuracy = get_accuracy(out[train_mask], labels[train_mask])

        gnn.eval()
        pred = gnn(features, adj)
        val_accuracy = get_accuracy(pred[val_mask], labels[val_mask])
        # print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}')

        if epoch == num_epochs - 1:
            if not converged:
                converged_epoch = epoch + 1

        # Check for convergence.
        if len(accuracies) >= 5:
            recent_accuracies = accuracies[-5:]
            # Check if the accuracy has not increased drastically within the last 5 epochs.
            if all((val_accuracy - a) <= 0.02 for a in recent_accuracies):
                # If not converegd already, mark it as converegd.
                if not converged:
                    converged_epoch = epoch - 4
                    converged = True
            else:
                converged = False  # Reset convergence flag if accuracy increased drastically.

        accuracies.append(val_accuracy)

    finish_time = time.time()
    elapsed_time = finish_time - start_time

    pred = gnn(features, adj)
    test_accuracy = get_accuracy(pred[test_mask], labels[test_mask])
    print('\n----------------------------\n')
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Training Time: {elapsed_time} seconds.')
    print(f'Converged Epoch: {converged_epoch}\n')

    return test_accuracy, elapsed_time, converged_epoch

def plot_metric(df, metric, ylabel, filename, splits, models, bar_width, x):
    plt.figure(figsize=(10, 6))

    for i, split in enumerate(splits):
        # Filter data for the split and ensure correct model order
        values = [df[(df['model'] == model) & (df['data_split'] == split)][metric].values[0] for model in models]
        plt.bar(x + i * bar_width, values, width=bar_width, label=split)

    plt.xticks(x + bar_width * (len(splits) - 1) / 2, models, rotation=45)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by Model and Data Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_cora_datasplit_graphs():
    csv_path = "../output/gcn-cora-datasplit-output.csv"
    output_dir = "../output/plots/cora/datasplit/"

    # Load the CSV file into a DataFrame.
    df = pd.read_csv(csv_path)

    # 🔍 Unique x-axis groups and categories
    models = df['model'].unique()
    splits = df['data_split'].unique()

    x = np.arange(len(models))  # the label locations

    # Plot all three metrics
    plot_metric(df, "accuracy", "Accuracy", output_dir + "accuracy_plot.png", splits, models, 0.2, x)
    plot_metric(df, "elapsed_time", "Elapsed Time (s)", output_dir + "elapsed_time_plot.png", splits, models, 0.2, x)
    plot_metric(df, "num_epochs_to_converge", "Epochs to Converge", output_dir + "epochs_to_converge_plot.png", splits, models, 0.2, x)

def plot_metrics_from_csv(csv_path, ouput_dir):
    # Load data
    df = pd.read_csv(csv_path)

    metrics = [
        ("accuracy", "Accuracy", "concat_accuracy_plot.png"),
        ("elapsed_time", "Elapsed Time (s)", "concat_elapsed_time_plot.png"),
        ("num_epochs_to_converge", "Epochs to Converge", "concat_epochs_plot.png")
    ]

    # For each metric, plot a line for each dataset
    for metric, ylabel, filename in metrics:
        plt.figure(figsize=(10, 6))

        for dataset in df['dataset'].unique():
            sub_df = df[df['dataset'] == dataset]
            # Sort by model name for consistent line shape
            sub_df = sub_df.sort_values(by='model')
            plt.plot(sub_df['model'], sub_df[metric], marker='o', label=dataset)

        plt.xticks(rotation=45, ha='right')
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by Model (per Dataset)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ouput_dir + filename)
        plt.close()
