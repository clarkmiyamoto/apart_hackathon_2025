import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict

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

def load_results():
    """Load all result files and extract data"""
    results_dir = 'results'
    data = defaultdict(list)
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.pt') and filename.startswith('results_'):
            params = parse_filename(filename)
            if params:
                width, depth, auxiliary, seed = params
                
                # Load the PyTorch file
                filepath = os.path.join(results_dir, filename)
                try:
                    result_data = torch.load(filepath, map_location='cpu')
                    
                    # The result_data is a list of tuples: (baseline_teacher, baseline_student, results_teacher, results_student)
                    # We want the results_student which contains the trained student's 1-hot accuracy
                    if isinstance(result_data, list) and len(result_data) > 0:
                        # Get the last result (most recent run)
                        last_result = result_data[-1]
                        if len(last_result) == 4:
                            baseline_teacher, baseline_student, results_teacher, results_student = last_result
                            
                            # Extract the 1-hot accuracy from results_student
                            if isinstance(results_student, dict) and 'acc_1hot' in results_student:
                                accuracy = results_student['acc_1hot']
                                data[auxiliary].append({
                                    'width': width,
                                    'depth': depth,
                                    'auxiliary': auxiliary,
                                    'seed': seed,
                                    'accuracy': float(accuracy)
                                })
                                print(f"Loaded {filename}: width={width}, auxiliary={auxiliary}, accuracy={accuracy:.4f}")
                            else:
                                print(f"No acc_1hot found in {filename}: {results_student}")
                        else:
                            print(f"Unexpected result structure in {filename}: {len(last_result)} elements")
                    else:
                        print(f"Unexpected data structure in {filename}")
                            
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return data

def create_scatter_plot(data):
    """Create scatter plot with width vs accuracy, different colors for auxiliaries"""
    plt.figure(figsize=(12, 8))
    
    # Define colors for different auxiliary values
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (auxiliary, points) in enumerate(data.items()):
        if not points:  # Skip if no data for this auxiliary
            continue
            
        widths = [p['width'] for p in points]
        accuracies = [p['accuracy'] for p in points]
        
        color = colors[i % len(colors)]
        plt.scatter(widths, accuracies, 
                   c=color, 
                   label=f'Auxiliary={auxiliary}', 
                   alpha=0.7, 
                   s=100)
    
    plt.xlabel('Width (Hidden Size)', fontsize=12)
    plt.ylabel('Trained Student 1-Hot Accuracy', fontsize=12)
    plt.title('Width vs Accuracy by Auxiliary Size', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('width_vs_accuracy_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Loading results...")
    data = load_results()
    
    if data:
        print(f"Loaded data for {len(data)} auxiliary sizes")
        for auxiliary, points in data.items():
            print(f"Auxiliary {auxiliary}: {len(points)} data points")
        
        create_scatter_plot(data)
    else:
        print("No data loaded. Check the file structure and data format.") 