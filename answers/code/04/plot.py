import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
# df = pd.read_csv('/home/jonas/Documents/University/5. Semester/Algorithm Engineering/Exam Assignments/Chapters/04/sorting_algorithms_running_times.csv')
df = pd.read_csv('/home/jonas/Documents/University/5. Semester/Algorithm Engineering/Exam Assignments/Chapters/04/speedup_array.csv')

print(df.columns)
# Plotting the data
plt.figure(figsize=(10, 6))

# Loop through each column (except for the first one 'threads') and plot the data
for column in df.columns[1:]:
    plt.plot(df['Array Size'], df[column], label=f' {column}', marker='o')

# Add titles and labels
plt.xscale('log')
plt.title('Speedup over std::sort at varying array sizes', fontsize=14)
# plt.title('Speedup over std::sort (Time vs. Array Size)', fontsize=14)
# plt.title('Performance of Sorting Algorithms (Time vs. Threads)', fontsize=14)
plt.xlabel('Array Size (in MB)', fontsize=12)
# plt.xlabel('Threads', fontsize=12)
# plt.ylabel('Time (seconds)', fontsize=12)
plt.ylabel('Speedup', fontsize=12)

# Show grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
