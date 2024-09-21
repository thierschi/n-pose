import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configure to adapt to your log-files
training_meta_file = '../../logs/2024-09-04-np/training_meta.csv'
output_path = '~/tmp/'

df_models = pd.read_csv(training_meta_file, skipinitialspace=True)

# Normalize the metrics and calculate the combined metric for best ovarll
scaler = MinMaxScaler()
df_models[['val_ATE_norm', 'val_AOE_norm']] = scaler.fit_transform(df_models[['val_ATE', 'val_AOE']])
df_models['combined_metric'] = df_models[['val_ATE_norm', 'val_AOE_norm']].mean(axis=1)

# Sort the DataFrames by each metric
sorted_by_val_ATE = df_models.sort_values(by='val_ATE')
sorted_by_val_AOE = df_models.sort_values(by='val_AOE')
sorted_by_combined_metric = df_models.sort_values(by='combined_metric')

# Export the sorted DataFrames to separate CSV files
sorted_by_val_ATE.to_csv(f'{output_path}sorted_by_val_ATE.csv', index=False)
sorted_by_val_AOE.to_csv(f'{output_path}sorted_by_val_AOE.csv', index=False)
sorted_by_combined_metric.to_csv(f'{output_path}sorted_by_combined_metric.csv', index=False)

# Print confirmation messages
print(f"All models sorted by val_ATE have been saved to {output_path}sorted_by_val_ATE.csv")
print(f"All models sorted by val_AOE have been saved to {output_path}sorted_by_val_AOE.csv")
print(f"All models sorted by combined metric have been saved to {output_path}sorted_by_combined_metric.csv")
