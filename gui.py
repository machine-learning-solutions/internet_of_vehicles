import tkinter as tk
from tkinter import ttk
import environment  # Assuming environment is a separate module


def start_simulation(power_algo, channel_algo, mode):
    print(f"Simulating with Power: {power_algo}, Channel: {channel_algo}, Mode: {mode}")
    # Call the environment setup method with parameters from the GUI
    environment.setup_environment(power_algo, channel_algo, mode)

root = tk.Tk()
root.title("Simulation Control Panel")
root.geometry("400x250")

# Use frames for better layout control
frame = ttk.Frame(root, padding="10")
frame.pack(fill='both', expand=True)

# Create and place dropdowns for selecting algorithms and modes with labels
power_var = tk.StringVar()
channel_var = tk.StringVar()
mode_var = tk.StringVar()

power_label = ttk.Label(frame, text="Power Allocation Algorithm:")
power_label.pack(pady=(10, 0))  # Add some vertical padding for spacing
power_dropdown = ttk.Combobox(frame, textvariable=power_var, values=['Proportional Power',])
power_dropdown.pack()

channel_label = ttk.Label(frame, text="Channel Allocation Algorithm:")
channel_label.pack(pady=(10, 0))  # Add some vertical padding for spacing
channel_dropdown = ttk.Combobox(frame, textvariable=channel_var, values=['Stable', 'Greedy', 'WUA'])
channel_dropdown.pack()

mode_label = ttk.Label(frame, text="Operation Mode:")
mode_label.pack(pady=(10, 0))  # Add some vertical padding for spacing
mode_dropdown = ttk.Combobox(frame, textvariable=mode_var, values=['Manual', 'Auto'])
mode_dropdown.pack()

# Button to start the simulation
start_button = ttk.Button(frame, text="Start Simulation", command=lambda: start_simulation(power_var.get(), channel_var.get(), mode_var.get()))
start_button.pack(pady=20)  # Add vertical padding to separate from dropdowns

root.mainloop()
