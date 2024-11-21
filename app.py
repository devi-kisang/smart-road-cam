from flask import Flask, render_template
import subprocess

app = Flask(__name__)

def get_accident_status():
    try:
        # Run the accident detection script and capture output
        result = subprocess.run(["python", "accident_detection.py"], capture_output=True, text=True)
        if "Accident happened" in result.stdout:
            return "Accident happened"
        else:
            return "No accident"
    except Exception as e:
        return f"Error: {e}"

def get_traffic_status():
    try:
        # Run the traffic monitoring script and capture output
        result = subprocess.run(["python", "traffic_detection.py"], capture_output=True, text=True)
        if "Traffic is normal" in result.stdout:
            return "Traffic is normal"
        else:
            return "Traffic is abnormal"
    except Exception as e:
        return f"Error: {e}"

@app.route('/')
def index():
    accident_status = get_accident_status()
    traffic_status = get_traffic_status()
    
    return render_template("index.html", accident_status=accident_status, traffic_status=traffic_status)

if __name__ == '__main__':
    app.run(debug=True)
