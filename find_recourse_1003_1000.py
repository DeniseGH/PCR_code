
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import recourse
import data_utils
import trainers
import os
from scm import SCM_PCR_linear
from scm import SCM_PCR_linear_estimated
from scm import SCM_PCR_linear_Test
from scm import SCM_PCR_linear_estimated_Test
import csv


# Function to print and log to a file simultaneously
def print_and_log(message, log_file):
    print(message)  # Print to console
    with open(log_file, 'a') as f:
        f.write(message + '\n')  # Write to file

def find_recourse_mlp_with_comparison(model, scmm_estimated, scmm_true, X_explain, constraints, epsilon, learning_rate):
    print_and_log("\n ##################### Entering find_recourse_MLP with comparison", log_file)
    hyperparams = {'lr': learning_rate, 'lambd_init': 1.0, 'decay_rate': 0.02, 'outer_iters': 50, 'inner_iters': 10, 'recourse_lr': 0.1}
    explainer = recourse.DifferentiableRecourse(model, hyperparams)

    print_and_log("Starting causal recourse computation with SCM_PCR_linear_estimated", log_file)
    
    interv_estimated, _, cost_recourse_estimated, _, _ = recourse.causal_recourse(X_explain, explainer, constraints, scm=scmm_estimated, epsilon=epsilon, robust=epsilon>0)

    print_and_log("Causal recourse computation with SCM_PCR_linear_estimated finished", log_file)
    

    if not isinstance(X_explain, torch.Tensor):
        X_explain = torch.tensor(X_explain)

    # Ensure interv_estimated is a torch.Tensor of shape (N, D)
    if not isinstance(interv_estimated, torch.Tensor):
        interv_estimated = torch.tensor(interv_estimated)

    # Check if X_explain and interv_estimated have the same shape (N, D)
    if X_explain.shape != interv_estimated.shape:
        raise ValueError("X_explain and interv_estimated must have the same shape (N, D).")


    X_results = scmm_true.counterfactual(X_explain, interv_estimated)
    preds = model.predict(X_results)

    # Print and log the counts
    print_and_log(f"Number of ones: {(preds == 1).sum()}, Number of zeros: {(preds == 0).sum()} out of the total {len(preds)}", log_file)


    # Now check validity with the true SCM (SCM_PCR_linear)
    print_and_log("Starting causal recourse validation with SCM_PCR_linear (true SCM)", log_file)
    _, recourse_valid_true, cost_recourse_true, _, _ = recourse.causal_recourse(
         X_explain, explainer, constraints, scm=scmm_true, epsilon=epsilon, robust=epsilon>0)

    print_and_log(f"Number of ones: {(recourse_valid_true == True).sum()}, Number of zeros: {(recourse_valid_true == False).sum()} out of the total {len(recourse_valid_true)}", log_file)

    print_and_log("Causal recourse validation with SCM_PCR_linear finished", log_file)


    # Comparison logic
    recourse_comparison_valid = (preds.astype(np.bool_) & recourse_valid_true.astype(np.bool_))

    true_greater_estimated = 0 
    estimated_greater_true = 0 

    # Check cost where both recourse are valid under the true SCM
    cost_comparison = np.full(recourse_comparison_valid.shape, np.nan)  # Initialize with NaN
    for i, valid in enumerate(recourse_comparison_valid):
        if valid: 
            if cost_recourse_true[i] <= cost_recourse_estimated[i]:
                # print_and_log(f"\nCost for instance {i} is lower or equal for true SCM: True cost: {cost_recourse_true[i]}, Estimated cost: {cost_recourse_estimated[i]}", log_file)
                estimated_greater_true = estimated_greater_true + 1
            else:
                # print_and_log(f"\nWarning: Cost for instance {i} is higher for true SCM: True cost: {cost_recourse_true[i]}, Estimated cost: {cost_recourse_estimated[i]}", log_file)
                true_greater_estimated = true_greater_estimated + 1
            cost_comparison[i] = cost_recourse_true[i] - cost_recourse_estimated[i]

    print_and_log(f"\nCost was lower or equal for true SCM compared to the estimated {estimated_greater_true} times ", log_file)
    print_and_log(f"\nCost was greater for true SCM compared to the estimated {true_greater_estimated} times ", log_file)

    print_and_log(f"Cost difference mean (true SCM - estimated SCM) where valid under both: {np.nanmean(cost_comparison)}", log_file)

    return (interv_estimated, preds.astype(np.bool_), cost_recourse_estimated, recourse_valid_true, cost_recourse_true, recourse_comparison_valid, cost_comparison) 


### SET THE DIRECTORY
DAY_AND_TIME = "1003_1130"
# Create a results folder in the current directory
current_dir = os.getcwd()  # Get the current working directory
result_folder = os.path.join(current_dir, "Results")
os.makedirs(result_folder, exist_ok=True)  # Create the folder if it doesn't exist

log_file = os.path.join(result_folder, f'results_log_{DAY_AND_TIME}.txt')
csv_file_path = os.path.join(result_folder, f'results_summary_{DAY_AND_TIME}.csv')


# Initialize CSV file and write the header
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write header
    csvwriter.writerow([
        'true_thetas', 'learning_rate', 'epsilon', 'N_explain', 'num_ones_train', 
        'num_zeros_train', 'num_ones_test', 'num_zeros_test', 'model_type', 'num_epochs', 
        'correct_preds', 'MLP accuracy', "X_explain_shape", 
        'Recourse_estimated_in_True_SCM', 'Recourse_in_True_SCM', 'cost_difference_mean',
        'accuracy_preds', 'accuracy_true', 'accuracy_comparison', 
    ])



example_type = input("Which example do you want to use? (test/paper): ").strip().lower()

# Check user input and initialize variables accordingly
if example_type == "test":
    # Test example
    true_thetas = [-0.99, 0.99, 0.99]  # From true SCM
    scmm_true = SCM_PCR_linear_Test()  # True SCM
    scmm_estimated = SCM_PCR_linear_estimated_Test()  # Estimated SCM
else:
    # Paper example (default if user doesn't input 'test')
    true_thetas = [-0.99, 0.05, 0.25]  # From true SCM
    scmm_true = SCM_PCR_linear()  # True SCM
    scmm_estimated = SCM_PCR_linear_estimated()  # Estimated SCM



model_type = 'MLP'
epsilon = 0.1
N_explain = 500
print_and_log(f"\nChoosen epsilon: {epsilon}", log_file)

X, Y, constraints = data_utils.process_data('linear', scmm_true)
X_train, Y_train, X_test, Y_test = data_utils.train_test_split(X, Y)


# Count the number of 1s and 0s in Y_train
num_ones_train = np.sum(Y_train == 1)
num_zeros_train = np.sum(Y_train == 0)

# Count the number of 1s and 0s in Y_test
num_ones_test = np.sum(Y_test == 1)
num_zeros_test = np.sum(Y_test == 0)

# Print the results
print_and_log(f"Y_train: {num_ones_train} ones, {num_zeros_train} zeros", log_file)
print_and_log(f"Y_test: {num_ones_test} ones, {num_zeros_test} zeros \n", log_file)



model = trainers.MLP
model = model(X_train.shape[-1], actionable_features=constraints['actionable'], actionable_mask=False)

X_train_tensor = torch.Tensor(X_train)
Y_train_tensor = torch.Tensor(Y_train).reshape(-1)  # Make it (80,)



# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # Use Adam optimizer

num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    logits = model(X_train_tensor)  # This is model.forward()

    # Compute loss
    loss = criterion(logits, Y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')


model.set_threshold(0.3)

# Now check predictions
preds = model.predict(X_test)

correct_preds = ((preds == Y_test).sum())
print_and_log(f"Number of correct predictions: {correct_preds}", log_file)
print_and_log(f"MLP accuracy: {correct_preds/len(Y_test)}", log_file)


# Sample N_explain number of misclassified points
id_neg = model.predict(X_test) == 0


# Ensure id_neg is a PyTorch tensor (if it's still a NumPy array)
if isinstance(id_neg, np.ndarray):
    id_neg = torch.tensor(id_neg)

# Assuming X_test is a PyTorch tensor
X_neg = X_test[id_neg.bool()]  # Get the negative examples
N_Explain = min(N_explain, X_neg.shape[0])  # Number of examples to explain

# Randomly select indices
id_explain = torch.randperm(X_neg.shape[0])[:N_Explain]  # Randomly permute and take the first N_Explain indices

# Get the corresponding indices from the original id_neg
id_neg_explain = torch.nonzero(id_neg)[id_explain].squeeze()  # Get the actual negative indices

# Extract the explanations
X_explain = X_neg[id_explain]

# Print and log
print_and_log(f"Shape of the samples that need explanation: {X_explain.shape}\n", log_file)


learning_rate_list = [1, 0.75, 0.5, 0.25, 0.1, 0.01]
print_and_log(f"learning rates we will examine{learning_rate_list}", log_file)

# Loop through learning rates and log results
for lr in learning_rate_list:
    print_and_log(f"Learning Rate: {lr}\n", log_file)

    # Run the recourse computation with comparison
    (interv_estimated, preds, cost_recourse_estimated,
    recourse_valid_true, cost_recourse_true, recourse_comparison_valid, cost_comparison) = find_recourse_mlp_with_comparison(
        model, scmm_estimated, scmm_true, X_explain, constraints, epsilon, lr
    )

    # Calculate accuracies
    accuracy_preds = sum(preds) / len(preds)
    accuracy_true = sum(recourse_valid_true) / len(recourse_valid_true)
    accuracy_comparison = sum(recourse_comparison_valid) / len(recourse_comparison_valid)

    # Calculate cost difference mean
    cost_difference_mean = np.nanmean(cost_comparison)
   
    # Log the information to the CSV file
    with open(csv_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            true_thetas, lr, epsilon,N_explain, num_ones_train, 
            num_zeros_train, num_ones_test, num_zeros_test, model_type, num_epochs, 
            correct_preds, correct_preds/len(Y_test), X_explain.shape, 
            sum(preds), sum(recourse_valid_true), cost_difference_mean,
            accuracy_preds, accuracy_true, accuracy_comparison
        ])

    # Also log to the regular text file for backward compatibility (optional)
    print_and_log(f"Accuracy of Recourse valid under estimated SCM in the True SCM: {accuracy_preds}", log_file)
    print_and_log(f"Accuracy of Recourse valid under true SCM: {accuracy_true}", log_file)
    print_and_log(f"Accuracy of Recourse valid under both SCMs: {accuracy_comparison}", log_file)
    print_and_log("\n\n-----------------------------------------------------------\n\n", log_file)

