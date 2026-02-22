import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_trajectory(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Load microphones.csv from the same directory
    mic_csv_path = os.path.join(os.path.dirname(csv_path), 'microphones.csv')
    df_mics = pd.read_csv(mic_csv_path) if os.path.exists(mic_csv_path) else None

    # Check if required columns exist
    required_columns = ['sx', 'sy', 'is_active']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain columns: {required_columns}")
        return

    # Separate active and inactive points
    active_points = df[df['is_active'] == 1]
    inactive_points = df[df['is_active'] == 0]

    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the entire trajectory as a single line first (optional, for continuity)
    plt.plot(df['sx'], df['sy'], color='gray', linestyle='-', linewidth=1, alpha=0.3)

    # Plot segments based on is_active status
    for i in range(len(df) - 1):
        x_vals = [df['sx'].iloc[i], df['sx'].iloc[i+1]]
        y_vals = [df['sy'].iloc[i], df['sy'].iloc[i+1]]
        
        if df['is_active'].iloc[i] == 1:
            plt.plot(x_vals, y_vals, color='red', linestyle='-', linewidth=2)
        else:
            plt.plot(x_vals, y_vals, color='gray', linestyle='--', linewidth=2)

    # Add dummy lines for legend
    plt.plot([], [], color='red', linestyle='-', linewidth=2, label='Siren ON')
    plt.plot([], [], color='gray', linestyle='--', linewidth=2, label='Siren OFF')

    # Mark start and end points
    plt.scatter(df['sx'].iloc[0], df['sy'].iloc[0], color='green', s=100, marker='s', label='Start', zorder=5)
    plt.scatter(df['sx'].iloc[-1], df['sy'].iloc[-1], color='purple', s=100, marker='X', label='End', zorder=5)

    # Plot microphones as blue dots
    if df_mics is not None:
        plt.scatter(df_mics['mx'], df_mics['my'], color='blue', s=80, marker='o', label='Microphones', zorder=6)
        for _, row in df_mics.iterrows():
            plt.annotate(row['mic_id'], (row['mx'], row['my']),
                         textcoords='offset points', xytext=(6, 4), fontsize=8, color='blue')
    else:
        plt.scatter(0, 0, color='blue', s=150, marker='o', label='Listener (0,0)', zorder=6)

    # Formatting
    plt.title(f'2D Trajectory: {os.path.basename(os.path.dirname(csv_path))}')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.axis('equal')  # Ensure aspect ratio is 1:1

    # Show plot
    plt.tight_layout()
    plt.savefig(f"{os.path.basename(os.path.dirname(csv_path))}_trajectory.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D trajectory from gt.csv")
    parser.add_argument("csv_path", help="Path to the gt.csv file")
    args = parser.parse_args()
    
    plot_trajectory(args.csv_path)
