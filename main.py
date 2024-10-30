# %% Import modules
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ray
from ray import train, tune
from dataset import MEGGraphs
from train import train_func, test_best_model

# %% Retrieve dataset
# Define name of .ds folder
'''
Note: if you want to load multiple files at once, add more filenames to the list. 
'''
filenames = ["sub-PT06ses04_ses-20181204_task-somatosensory_run-06_meg.ds"]

# Define the duration of the subepochs and the amount of overlap between them
duration = 60
overlap = 0

# Define variables for node and edge features 
conn_method = 'pli'
freqs = {'fmin': 1, 
         'fmax': 40, 
         'freqs': np.linspace(1, 40, (40 - 1) * 4 + 1)}

dataset = MEGGraphs(root="data\\", 
                    filenames=filenames, 
                    duration=duration,
                    overlap=overlap,
                    conn_method=conn_method,
                    freqs=freqs)

# %% Split dataset in train and test set
labels = []
for data in dataset:
    labels.append(data.y)

dataset_train, dataset_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42, stratify=labels)

# %% Training
ray.shutdown()
ray.init(_temp_dir=r"D:\GNN\ray_temp")
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

search_space = {
    'hidden_channels': tune.choice([16, 32, 64, 128]),
    'lr': tune.loguniform(1e-5, 1e-1),
    'batch_size': tune.choice([2, 4, 6, 8])
}

run_config = train.RunConfig(
    storage_path=r"D:\GNN\ray_results"
)               

tune_config = tune.TuneConfig(
    metric='val_accuracy',
    mode='max',
    num_samples=5
)

tuner = tune.Tuner(
    tune.with_parameters(train_func, dataset=dataset, dataset_train=dataset_train, y_train=y_train),
    param_space=search_space,
    tune_config=tune_config,
    run_config=run_config
)

results = tuner.fit()

# %% Retrieve best model parameters
print(results.get_dataframe())
best_result = results.get_best_result()
best_result_che = best_result.metrics['val_accuracy']
best_params = best_result.config
print(best_params)

# %% Plot results
# Plot losses
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(best_result.metrics_dataframe['training_iteration'], best_result.metrics_dataframe['train_loss'], label='Train loss')
plt.plot(best_result.metrics_dataframe['training_iteration'], best_result.metrics_dataframe['val_loss'], label='Validation loss')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Training and Validation Loss per Epoch')
plt.grid(True)

# Plot accuracies 
plt.subplot(122)
plt.plot(best_result.metrics_dataframe['training_iteration'], best_result.metrics_dataframe['train_accuracy'], label='Train accuracy')
plt.plot(best_result.metrics_dataframe['training_iteration'], best_result.metrics_dataframe['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Training and Validation Accuracy per Epoch')
plt.grid(True)

# %%
print(best_result.metrics_dataframe)

# %% Test best model on test dataset
test_best_model(best_result, dataset, dataset_test)