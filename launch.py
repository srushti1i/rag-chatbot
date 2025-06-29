import subprocess, time
from pyngrok import ngrok, conf

# Kill previous runs if any
!pkill streamlit
ngrok.kill()

# Optional: Set ngrok auth token here (if not already done)
# !ngrok config add-authtoken YOUR_TOKEN_HERE

# Start Streamlit
print("🚀 Starting Streamlit...")
process = subprocess.Popen(["streamlit", "run", "app.py"])
time.sleep(5)  # Give Streamlit a moment

# Connect ngrok tunnel
try:
    public_url = ngrok.connect(8501)
    print(f"✅ Your app is live at: {public_url}")
except Exception as e:
    print("⚠️ ngrok connect failed. Trying manual backup.")
    !ngrok http 8501 &
    time.sleep(3)
    !curl -s http://localhost:4040/api/tunnels | grep -Eo 'https://[0-9a-z\-]+\.ngrok-free\.app'

