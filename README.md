📈 Real-Time Anomaly Detection Dashboard
A fully interactive web application built with Streamlit that simulates and visualizes a live time-series data stream, allowing for real-time anomaly detection using multiple algorithms.

➡️ [View the Live Demo Here](https://app-anomaly-dashboard-yxhdp6vafbuveg5rucbtrh.streamlit.app/)

It's highly recommended to add a screenshot or an animated GIF of your app in action here.

✨ Key Features
Live Data Simulation: Generates a continuous stream of data with configurable drift and random shocks to simulate real-world conditions.

Multiple Detection Algorithms: Switch between three distinct anomaly detection methods on the fly:

Z-Score: Best for data that follows a normal distribution (a bell curve).

Median Absolute Deviation (MAD): A robust method that is less sensitive to extreme outliers in the data.

Isolation Forest: A modern, machine learning-based approach that is efficient and does not assume a specific data distribution.

Interactive Controls: Dynamically adjust the rolling window size and algorithm-specific parameters (like Z-Score Threshold or Contamination) and see the impact in real-time.

Persistent Visualization: The complete history of the data stream and all detected anomalies remain on the chart, even when the simulation is paused.

Live Anomaly Log: A running log in the sidebar provides a timestamp and the value for every anomaly as it's detected.

🛠️ Tech Stack
Streamlit: For the web framework and interactive UI.

Plotly: For creating rich, interactive, and real-time data visualizations.

Pandas & NumPy: For efficient data manipulation and numerical calculations.

Scikit-learn: For the Isolation Forest machine learning algorithm.

🚀 How to Run Locally
Prerequisites: Ensure you have Python 3.8+ installed.

Clone the Repository:

git clone [https://github.com/Moqo12/streamlit-anomaly-dashboard.git](https://github.com/Moqo12/streamlit-anomaly-dashboard.git)
cd streamlit-anomaly-dashboard

Set up a Virtual Environment (Recommended):

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

Run the App:

streamlit run app.py

The application will open in your default web browser.

☁️ Deployment
This application is deployed and live on Streamlit Community Cloud.

Link your GitHub repository and deploy.
