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
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return data
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.pt'):
            params = parse_filename(filename)
            if params:
                width, depth, auxiliary, seed = params
                filepath = os.path.join(results_dir, filename)
                
                try:
                    # Load the results
                    results = torch.load(filepath, map_location='cpu')
                    
                    # The data structure is a list containing a tuple of 4 dictionaries
                    # We want the last dictionary (student results) which contains acc_1hot
                    if isinstance(results, list) and len(results) > 0:
                        # Get the tuple from the list
                        result_tuple = results[0]
                        if isinstance(result_tuple, tuple) and len(result_tuple) == 4:
                            # The last element is the student results
                            student_results = result_tuple[3]
                            
                            # Extract the 1-hot accuracy
                            if isinstance(student_results, dict) and 'acc_1hot' in student_results:
                                accuracy = student_results['acc_1hot']
                                data['widths'].append(width)
                                data['auxiliaries'].append(auxiliary)
                                data['depths'].append(depth)
                                data['accuracies'].append(accuracy)
                                data['seeds'].append(seed)
                            else:
                                print(f"No acc_1hot found in {filename}: {student_results}")
                        else:
                            print(f"Unexpected result_tuple structure in {filename}")
                    else:
                        print(f"Unexpected results structure in {filename}")
                        
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return data

def create_scatter_plot_with_averages():
    """Create scatter plot with width vs accuracy, auxiliary as colors, with average lines"""
    data = load_results()
    
    if not data['widths']:
        print("No data found!")
        return
    
    # Convert to numpy arrays
    widths = np.array(data['widths'])
    auxiliaries = np.array(data['auxiliaries'])
    accuracies = np.array(data['accuracies'])
    
    # Get unique auxiliary values and sort them
    unique_auxiliaries = sorted(set(auxiliaries))
    
    # Create aesthetic color palette and shapes
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']  # Blue, Purple, Orange, Red, Deep Purple
    shapes = ['o', 's', '^', 'D', 'v']  # Circle, Square, Triangle, Diamond, Inverted Triangle
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual points with different shapes and colors
    for i, auxiliary in enumerate(unique_auxiliaries):
        mask = auxiliaries == auxiliary
        color = colors[i % len(colors)]
        shape = shapes[i % len(shapes)]
        
        # Plot individual points with transparency
        plt.scatter(widths[mask], accuracies[mask], 
                   c=color, marker=shape, s=120, alpha=0.6, 
                   edgecolors=color, linewidth=0.5,
                   label=f'Auxiliary={auxiliary}')
        
        # Calculate and plot average line
        unique_widths = sorted(set(widths[mask]))
        avg_accuracies = []
        avg_widths = []
        
        for width in unique_widths:
            width_mask = (widths == width) & (auxiliaries == auxiliary)
            if np.sum(width_mask) > 0:
                avg_acc = np.mean(accuracies[width_mask])
                avg_accuracies.append(avg_acc)
                avg_widths.append(width)
        
        if len(avg_widths) > 1:
            # Sort by width for proper line connection
            sorted_indices = np.argsort(avg_widths)
            sorted_widths = np.array(avg_widths)[sorted_indices]
            sorted_accs = np.array(avg_accuracies)[sorted_indices]
            
            plt.plot(sorted_widths, sorted_accs, color=color, linewidth=3, alpha=0.9)
    
    # Customize the plot
    plt.xscale('log')
    plt.xlabel('Hidden Width', fontsize=14, fontweight='bold')
    plt.ylabel('Trained Student 1-Hot Accuracy', fontsize=14, fontweight='bold')
    plt.title('Width vs Accuracy by Auxiliary Size', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Order legend by auxiliary size
    handles, labels = plt.gca().get_legend_handles_labels()
    # Sort by auxiliary number
    legend_data = [(int(label.split('=')[1]), handle, label) for handle, label in zip(handles, labels)]
    legend_data.sort(key=lambda x: x[0])
    handles = [item[1] for item in legend_data]
    labels = [item[2] for item in legend_data]
    
    plt.legend(handles, labels, title='Auxiliary Size', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('width_vs_accuracy_auxiliary_aesthetic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nData Summary:")
    print(f"Total data points: {len(widths)}")
    print(f"Unique widths: {sorted(set(widths))}")
    print(f"Unique auxiliaries: {unique_auxiliaries}")
    
    for auxiliary in unique_auxiliaries:
        mask = auxiliaries == auxiliary
        mean_acc = np.mean(accuracies[mask])
        std_acc = np.std(accuracies[mask])
        count = np.sum(mask)
        print(f"Auxiliary {auxiliary}: Mean={mean_acc:.3f}, Std={std_acc:.3f}, Count={count}")

if __name__ == "__main__":
    create_scatter_plot_with_averages() 