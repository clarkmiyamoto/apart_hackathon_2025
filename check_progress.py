import itertools

# Define the parameter ranges from your notebook
hiddens = [2 ** j for j in range(6, 13)]  # [64, 128, 256, 512, 1024, 2048, 4096]
depths = [1, 2, 3, 5]
auxiliaries = [3, 10, 50, 100, 1000]

# Create all possible combinations
itt = list(itertools.product(hiddens, depths, auxiliaries))

# Completed combinations (extracted from file names)
completed = [
    (1024, 1, 3),
    (128, 1, 10),
    (128, 1, 100),
    (128, 1, 3),
    (128, 1, 50),
    (256, 1, 10),
    (256, 1, 100),
    (256, 1, 3),
    (256, 1, 50),
    (64, 1, 10),
    (64, 1, 100),
    (64, 1, 1000),
    (64, 1, 3),
    (64, 1, 50),
    (64, 2, 10),
    (64, 2, 100),
    (64, 2, 1000),
    (64, 2, 3),
    (64, 2, 50),
    (64, 3, 10),
    (64, 3, 3),
    (64, 3, 50),
]

print(f"Total possible combinations: {len(itt)}")
print(f"Completed combinations: {len(completed)}")
print(f"Remaining combinations: {len(itt) - len(completed)}")
print()

# Find indices of completed combinations (1-based indexing)
completed_indices = []
for i, combo in enumerate(itt):
    if combo in completed:
        completed_indices.append(i + 1)  # 1-based indexing

print("COMPLETED INDICES (1-based):")
print(f"itt[{', '.join(map(str, sorted(completed_indices)))}]")
print()

print("COMPLETED COMBINATIONS WITH INDICES:")
for idx in sorted(completed_indices):
    hidden, depth, auxiliary = itt[idx - 1]  # Convert back to 0-based
    print(f"itt[{idx}]: Hidden={hidden}, Depth={depth}, Auxiliary={auxiliary}")

print("\n" + "="*50 + "\n")

# Find indices of remaining combinations
remaining_indices = []
for i, combo in enumerate(itt):
    if combo not in completed:
        remaining_indices.append(i + 1)  # 1-based indexing

print("REMAINING INDICES (1-based):")
print(f"itt[{', '.join(map(str, sorted(remaining_indices)))}]")

print(f"\nProgress: {len(completed)}/{len(itt)} = {len(completed)/len(itt)*100:.1f}% complete") 