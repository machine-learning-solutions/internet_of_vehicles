import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import environment  # Assuming environment is a separate module

class SimulationApp(toga.App):
    def startup(self):
        # Create the main window
        self.main_window = toga.MainWindow(title='Simulation Control Panel')

        # Power Allocation Algorithm
        power_label = toga.Label('Power Allocation Algorithm:', style=Pack(padding=(0, 5)))
        self.power_dropdown = toga.Selection(items=['Proportional', 'BLCA'], style=Pack(flex=1))

        # Channel Allocation Algorithm
        channel_label = toga.Label('Channel Allocation Algorithm:', style=Pack(padding=(0, 5)))
        self.channel_dropdown = toga.Selection(items=['Stable', 'Greedy', 'WUA'], style=Pack(flex=1))

        # Operation Mode
        mode_label = toga.Label('Operation Mode:', style=Pack(padding=(0, 5)))
        self.mode_dropdown = toga.Selection(items=['Indirect via Leader', 'Indirect without Leader', 'Direct'], style=Pack(flex=1))

        # Number of Episodes
        episodes_label = toga.Label('Number of Episodes:', style=Pack(padding=(0, 5)))
        self.episodes_input = toga.TextInput(placeholder='Enter number of episodes', style=Pack(flex=1))

        # Start Simulation Button
        start_button = toga.Button('Start Simulation', on_press=self.start_simulation, style=Pack(padding=10))

        # Arrange widgets in boxes
        power_box = toga.Box(children=[power_label, self.power_dropdown], style=Pack(direction=ROW, padding=5))
        channel_box = toga.Box(children=[channel_label, self.channel_dropdown], style=Pack(direction=ROW, padding=5))
        mode_box = toga.Box(children=[mode_label, self.mode_dropdown], style=Pack(direction=ROW, padding=5))
        episodes_box = toga.Box(children=[episodes_label, self.episodes_input], style=Pack(direction=ROW, padding=5))

        # Main content box
        content_box = toga.Box(
            children=[power_box, channel_box, mode_box, episodes_box, start_button],
            style=Pack(direction=COLUMN, padding=10)
        )

        # Set the content of the main window
        self.main_window.content = content_box
        self.main_window.show()

    def start_simulation(self, widget):
        try:
            # Retrieve values from inputs
            power_algo = self.power_dropdown.value
            channel_algo = self.channel_dropdown.value
            mode = self.mode_dropdown.value
            num_episodes = int(self.episodes_input.value)

            print(f"Simulating with Power: {power_algo}, Channel: {channel_algo}, Mode: {mode}, Episodes: {num_episodes}")
            # Call the environment setup method with parameters from the GUI
            environment.setup_environment(power_algo, channel_algo, mode, num_episodes)
        except ValueError:
            # Show an alert dialog if the number of episodes is not an integer
            self.main_window.error_dialog('Input Error', 'Number of episodes must be an integer.')

def main():
    return SimulationApp('Simulation Control Panel', 'org.example.simulation')

if __name__ == '__main__':
    app = SimulationApp('Simulation Control Panel', 'org.example.simulation')
    app.main_loop()
