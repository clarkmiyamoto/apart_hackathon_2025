import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict
import matplotlib.colors as mcolors

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
                                data[depth].append({
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

def create_scatter_plot_width_depth(data):
    """Create scatter plot with width vs accuracy, depth as colors and shapes, and average lines"""
    plt.figure(figsize=(14, 10))
    
    # Get all depth values and create color/shape mapping
    depth_values = sorted(data.keys())
    
    # Create aesthetic color palette and shape mapping
    colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Beautiful blues, purples, oranges, reds
    markers_list = ['o', 's', '^', 'D']  # Circle, square, triangle, diamond
    
    # Create color and shape mapping based on depth value
    depth_to_color = {}
    depth_to_marker = {}
    for i, depth in enumerate(depth_values):
        color_idx = i % len(colors_list)
        marker_idx = i % len(markers_list)
        depth_to_color[depth] = colors_list[color_idx]
        depth_to_marker[depth] = markers_list[marker_idx]
    
    # Plot in order of depth (smallest to largest) for proper legend ordering
    for depth in depth_values:
        points = data[depth]
        if not points:  # Skip if no data for this depth
            continue
            
        widths = [p['width'] for p in points]
        accuracies = [p['accuracy'] for p in points]
        
        color = depth_to_color[depth]
        marker = depth_to_marker[depth]
        
        # Plot individual points with transparency
        plt.scatter(widths, accuracies, 
                   c=color, 
                   marker=marker,
                   label=f'Depth={depth}', 
                   alpha=0.6,  # Translucent dots
                   s=120,  # Slightly larger dots
                   edgecolors=color,
                   linewidth=0.5)
        
        # Calculate and plot average line
        # Group by width and calculate mean accuracy for each width
        width_groups = defaultdict(list)
        for point in points:
            width_groups[point['width']].append(point['accuracy'])
        
        avg_widths = []
        avg_accuracies = []
        for width in sorted(width_groups.keys()):
            avg_widths.append(width)
            avg_accuracies.append(np.mean(width_groups[width]))
        
        # Plot average line with same marker but solid
        plt.plot(avg_widths, avg_accuracies, 
                color=color, 
                linewidth=3, 
                alpha=0.9,
                linestyle='-',
                marker=marker,
                markersize=10,
                markeredgecolor=color,
                markeredgewidth=1.5)
    
    plt.xlabel('Width (Hidden Size)', fontsize=14)
    plt.ylabel('Trained Student 1-Hot Accuracy', fontsize=14)
    plt.title('Width vs Accuracy by Depth (with Average Lines)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.ylim(0, 1)
    plt.xscale('log')  # Use log scale for width to better show the relationship
    plt.xticks([64, 128, 256, 512, 1024, 2048], ['64', '128', '256', '512', '1024', '2048'])
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('width_vs_accuracy_by_depth.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_average_statistics(data):
    """Print average statistics for each depth"""
    print("\n=== AVERAGE ACCURACY BY WIDTH AND DEPTH ===")
    
    for depth in sorted(data.keys()):
        points = data[depth]
        if not points:
            continue
            
        print(f"\nDepth {depth}:")
        
        # Group by width
        width_groups = defaultdict(list)
        for point in points:
            width_groups[point['width']].append(point['accuracy'])
        
        for width in sorted(width_groups.keys()):
            accuracies = width_groups[width]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"  Width {width}: Mean={mean_acc:.4f}, Std={std_acc:.4f}, N={len(accuracies)}")

if __name__ == "__main__":
    print("Loading results...")
    data = load_results()
    
    if data:
        print(f"Loaded data for {len(data)} depths")
        for depth, points in data.items():
            print(f"Depth {depth}: {len(points)} data points")
        
        print_average_statistics(data)
        create_scatter_plot_width_depth(data)
    else:
        print("No data loaded. Check the file structure and data format.") 