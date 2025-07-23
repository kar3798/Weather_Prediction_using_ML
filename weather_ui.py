import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from weather_data_parser import get_latest_weather

class WeatherApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Live Weather Data")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        # Labels
        self.temp_label = QLabel("Temperature: Loading...", self)
        self.humidity_label = QLabel("Humidity: Loading...", self)
        self.pressure_label = QLabel("Pressure: Loading...", self)
        self.wind_label = QLabel("Wind Speed: Loading...", self)
        self.status_label = QLabel("", self)

        # Refresh Button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.load_weather)

        # Add widgets to layout
        layout.addWidget(self.temp_label)
        layout.addWidget(self.humidity_label)
        layout.addWidget(self.pressure_label)
        layout.addWidget(self.wind_label)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        # Load weather data initially
        self.load_weather()

    def load_weather(self):
        weather = get_latest_weather()

        if weather:
            self.temp_label.setText(f"Temperature: {weather['Temperature']} °C")
            self.humidity_label.setText(f"Humidity: {weather['Humidity']}%")
            self.pressure_label.setText(f"Pressure: {weather['Pressure']} hPa")
            self.wind_label.setText(f"Wind Speed: {weather['Wind Speed']} m/s")
            self.status_label.setText("Updated successfully")
        else:
            self.status_label.setText("No data available for today.")

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WeatherApp()
    window.show()
    sys.exit(app.exec_())

