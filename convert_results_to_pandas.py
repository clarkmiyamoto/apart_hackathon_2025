import torch
import pandas as pd
import os
import re
from collections import defaultdict
import numpy as np

def parse_filename(filename):
    """Parse the filename to extract parameters"""
    # Remove .pt extension
    name = filename.replace('.pt', '')
    
    # Extract parameters using regex
    pattern = r'results_Hidden(\d+)_Depth(\d+)_Auxiliary(\d+)_seed(\d+)'
    match = re.match(pattern, name)
    
    if match:
        width = int(match.group(1))
        depth = int(match.group(2))
        auxiliary = int(match.group(3))
        seed = int(match.group(4))
        return width, depth, auxiliary, seed
    return None

def load_results_to_dataframe():
    """Load all result files and convert to pandas DataFrame"""
    results_dir = 'results'
    data_list = []
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return pd.DataFrame()
    
    print("Loading results from PyTorch files...")
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.pt') and filename.startswith('results_'):
            params = parse_filename(filename)
            if params:
                width, depth, auxiliary, seed = params
                filepath = os.path.join(results_dir, filename)
                
                try:
                    # Load the PyTorch file
                    results = torch.load(filepath, map_location='cpu')
                    
                    # The data structure is a list containing tuples of 4 dictionaries
                    # Each tuple is (baseline_teacher, baseline_student, results_teacher, results_student)
                    # We want the results_student which contains acc_1hot
                    if isinstance(results, list) and len(results) > 0:
                        # Get the last tuple from the list (most recent run)
                        result_tuple = results[-1]
                        if isinstance(result_tuple, tuple) and len(result_tuple) == 4:
                            # The last element is the student results
                            student_results = result_tuple[3]
                            
                            # Extract the 1-hot accuracy
                            if isinstance(student_results, dict) and 'acc_1hot' in student_results:
                                accuracy = float(student_results['acc_1hot'])
                                
                                # Create a row for the DataFrame
                                row = {
                                    'filename': filename,
                                    'width': width,
                                    'depth': depth,
                                    'auxiliary': auxiliary,
                                    'seed': seed,
                                    'accuracy': accuracy
                                }
                                
                                # Add any additional metrics from student_results if available
                                for key, value in student_results.items():
                                    if key != 'acc_1hot':  # Already added
                                        if isinstance(value, (int, float)):
                                            row[f'student_{key}'] = value
                                        elif isinstance(value, torch.Tensor):
                                            row[f'student_{key}'] = float(value.item())
                                        else:
                                            row[f'student_{key}'] = str(value)
                                
                                data_list.append(row)
                                print(f"Loaded {filename}: width={width}, depth={depth}, auxiliary={auxiliary}, seed={seed}, accuracy={accuracy:.4f}")
                            else:
                                print(f"No acc_1hot found in {filename}: {student_results}")
                        else:
                            print(f"Unexpected result_tuple structure in {filename}")
                    else:
                        print(f"Unexpected results structure in {filename}")
                        
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    # Create DataFrame
    if data_list:
        df = pd.DataFrame(data_list)
        
        # Sort by width, depth, auxiliary, seed for better organization
        df = df.sort_values(['width', 'depth', 'auxiliary', 'seed']).reset_index(drop=True)
        
        print(f"\nSuccessfully loaded {len(df)} data points")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Unique widths: {sorted(df['width'].unique())}")
        print(f"Unique depths: {sorted(df['depth'].unique())}")
        print(f"Unique auxiliary sizes: {sorted(df['auxiliary'].unique())}")
        print(f"Number of seeds per configuration: {df.groupby(['width', 'depth', 'auxiliary']).size().describe()}")
        
        return df
    else:
        print("No data loaded. Check the file structure and data format.")
        return pd.DataFrame()

def save_dataframe(df, filename='results_dataframe.csv'):
    """Save the DataFrame to CSV file"""
    if not df.empty:
        df.to_csv(filename, index=False)
        print(f"\nDataFrame saved to {filename}")
        
        # Also save as pickle for faster loading later
        pickle_filename = filename.replace('.csv', '.pkl')
        df.to_pickle(pickle_filename)
        print(f"DataFrame also saved as pickle to {pickle_filename}")
    else:
        print("No data to save.")

def print_sample_data(df, n=5):
    """Print a sample of the data"""
    if not df.empty:
        print(f"\nSample data (first {n} rows):")
        print(df.head(n).to_string())
        
        print(f"\nData types:")
        print(df.dtypes)
        
        print(f"\nBasic statistics for numerical columns:")
        print(df.describe())

if __name__ == "__main__":
    # Load and convert the data
    df = load_results_to_dataframe()
    
    if not df.empty:
        # Print sample data
        print_sample_data(df)
        
        # Save to CSV and pickle
        save_dataframe(df)
        
        # Print some useful queries
        print(f"\nExample queries you can run:")
        print(f"df[df['auxiliary'] == 3]  # Filter by auxiliary size")
        print(f"df.groupby('width')['accuracy'].mean()  # Average accuracy by width")
        print(f"df.groupby(['width', 'depth'])['accuracy'].agg(['mean', 'std', 'count'])  # Detailed statistics")
        print(f"df.pivot_table(index='width', columns='auxiliary', values='accuracy', aggfunc='mean')  # Pivot table")
    else:
        print("No data was loaded. Please check the results folder and file structure.") 