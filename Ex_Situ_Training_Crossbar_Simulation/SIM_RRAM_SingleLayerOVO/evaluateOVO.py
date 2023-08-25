import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
def evaluation_Plots():
    csv_files = [('OVO_results.csv', '#C26DBC', 'Single-Layer'), ]
    
    if not os.path.exists('Figures'):
        os.makedirs('Figures')

    fig, ax = plt.subplots(figsize=(10, 6))

    for file_name, color, label in csv_files:
        csv_file = f'results_data/{file_name}'

        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)

            # Count the number of each deviation value in both columns
            failure_counts = df['Failure Deviation'].value_counts()

            # Combine the two DataFrames into a single one
            counts_df = pd.DataFrame({'Failure': failure_counts}).fillna(0)

            # Sort the DataFrame by the index (which is the deviation values)
            counts_df.sort_index(inplace=True)

            # Calculate percentage of failures at each deviation
            counts_df['Failure Percentage'] = (counts_df['Failure'] / counts_df['Failure'].sum()) * 100

            # Calculate cumulative percentage of failures
            counts_df['Cumulative Failure Percentage'] = counts_df['Failure Percentage'].cumsum()

            # Insert a row at the start for the deviation that is one less than the smallest deviation
            counts_df.loc[counts_df.index.min() - 1] = 0
            counts_df.sort_index(inplace=True)

            # Create a line plot
            ax.plot(counts_df.index, counts_df['Cumulative Failure Percentage'], color=color, marker='o', label=label)

            # Annotate cumulative percentage at each point as an integer
            for i, txt in enumerate(counts_df['Cumulative Failure Percentage']):
                ax.annotate(int(txt), (counts_df.index[i], txt), va='bottom', ha='center')

        else:
            print(f"Data File {file_name} Not Found")

    ax.set_title('Failure Rate vs. Conductance Control Deviation')
    ax.set_ylabel('Failure Rate')
    ax.set_xlabel('Conductance Control Deviation')
    ax.legend()

    # Fill under the line
    for file_name, color, label in csv_files:
        csv_file = f'results_data/{file_name}'
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
            # Calculate and print mean and mode for Success and Failure Deviation
            success_mean = df['Success Deviation'].mean()
            success_mode = df['Success Deviation'].mode()[0]
            failure_mean = df['Failure Deviation'].mean()
            failure_mode = df['Failure Deviation'].mode()[0]
            print(f"For {label}:\nSuccess Mean: {success_mean}, Success Mode: {success_mode}\nFailure Mean: {failure_mean}, Failure Mode: {failure_mode}")
            failure_counts = df['Failure Deviation'].value_counts()
            counts_df = pd.DataFrame({'Failure': failure_counts}).fillna(0)
            counts_df.sort_index(inplace=True)
            counts_df['Failure Percentage'] = (counts_df['Failure'] / counts_df['Failure'].sum()) * 100
            counts_df['Cumulative Failure Percentage'] = counts_df['Failure Percentage'].cumsum()
            counts_df.loc[counts_df.index.min() - 1] = 0
            counts_df.sort_index(inplace=True)
            ax.fill_between(counts_df.index, counts_df['Cumulative Failure Percentage'], color=color, alpha=0.4)

    # Set the background color and grid color
    ax.set_facecolor('#ededed')
    ax.grid(color='white')

    # Set y limits
    max_percentage = max([counts_df['Cumulative Failure Percentage'].max() for file_name, color, label in csv_files])
    ax.set_ylim(0, 1.1 * max_percentage)

    plt.savefig('Figures/SLP_percentage.png',dpi=300)  
    plt.show()




    
def main():
     evaluation_Plots()

if __name__ == "__main__":
    main()