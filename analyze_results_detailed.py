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
    data = defaultdict(lambda: defaultdict(list))  # depth -> auxiliary -> data points
    
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
                                data[depth][auxiliary].append({
                                    'width': width,
                                    'depth': depth,
                                    'auxiliary': auxiliary,
                                    'seed': seed,
                                    'accuracy': float(accuracy)
                                })
                                print(f"Loaded {filename}: width={width}, depth={depth}, auxiliary={auxiliary}, accuracy={accuracy:.4f}")
                            else:
                                print(f"No acc_1hot found in {filename}: {results_student}")
                        else:
                            print(f"Unexpected result structure in {filename}: {len(last_result)} elements")
                    else:
                        print(f"Unexpected data structure in {filename}")
                            
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return data

def create_scatter_plots(data):
    """Create scatter plots for each depth"""
    depths = sorted(data.keys())
    
    # Create subplots for each depth
    fig, axes = plt.subplots(1, len(depths), figsize=(6*len(depths), 8))
    if len(depths) == 1:
        axes = [axes]
    
    # Define colors for different auxiliary values
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, depth in enumerate(depths):
        ax = axes[i]
        depth_data = data[depth]
        
        for j, (auxiliary, points) in enumerate(depth_data.items()):
            if not points:  # Skip if no data for this auxiliary
                continue
                
            widths = [p['width'] for p in points]
            accuracies = [p['accuracy'] for p in points]
            
            color = colors[j % len(colors)]
            ax.scatter(widths, accuracies, 
                      c=color, 
                      label=f'Auxiliary={auxiliary}', 
                      alpha=0.7, 
                      s=100)
        
        ax.set_xlabel('Width (Hidden Size)', fontsize=12)
        ax.set_ylabel('Trained Student 1-Hot Accuracy', fontsize=12)
        ax.set_title(f'Depth {depth}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)  # Set y-axis from 0 to 1 for accuracy
    
    plt.tight_layout()
    plt.savefig('width_vs_accuracy_by_depth.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_combined_plot(data):
    """Create a combined scatter plot with all depths"""
    plt.figure(figsize=(14, 10))
    
    # Define colors for different auxiliary values
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v']  # Different markers for different depths
    
    for depth_idx, depth in enumerate(sorted(data.keys())):
        depth_data = data[depth]
        
        for aux_idx, (auxiliary, points) in enumerate(depth_data.items()):
            if not points:  # Skip if no data for this auxiliary
                continue
                
            widths = [p['width'] for p in points]
            accuracies = [p['accuracy'] for p in points]
            
            color = colors[aux_idx % len(colors)]
            marker = markers[depth_idx % len(markers)]
            
            plt.scatter(widths, accuracies, 
                       c=color, 
                       marker=marker,
                       s=120,
                       alpha=0.7,
                       label=f'Depth{depth}_Aux{auxiliary}')
    
    plt.xlabel('Width (Hidden Size)', fontsize=14)
    plt.ylabel('Trained Student 1-Hot Accuracy', fontsize=14)
    plt.title('Width vs Accuracy by Depth and Auxiliary Size', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    plt.savefig('width_vs_accuracy_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(data):
    """Print summary statistics for the data"""
    print("\n=== SUMMARY STATISTICS ===")
    
    for depth in sorted(data.keys()):
        print(f"\nDepth {depth}:")
        depth_data = data[depth]
        
        for auxiliary in sorted(depth_data.keys()):
            points = depth_data[auxiliary]
            if points:
                accuracies = [p['accuracy'] for p in points]
                widths = [p['width'] for p in points]
                
                print(f"  Auxiliary {auxiliary}: {len(points)} points")
                print(f"    Widths: {sorted(set(widths))}")
                print(f"    Accuracy - Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")
                print(f"    Accuracy - Min: {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}")

if __name__ == "__main__":
    print("Loading results...")
    data = load_results()
    
    if data:
        print(f"Loaded data for {len(data)} depths")
        for depth, depth_data in data.items():
            print(f"Depth {depth}: {len(depth_data)} auxiliary sizes")
            for auxiliary, points in depth_data.items():
                print(f"  Auxiliary {auxiliary}: {len(points)} data points")
        
        print_summary_statistics(data)
        create_scatter_plots(data)
        create_combined_plot(data)
    else:
        print("No data loaded. Check the file structure and data format.") 