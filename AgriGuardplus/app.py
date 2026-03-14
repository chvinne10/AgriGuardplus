import os
import cv2
import mysql.connector
from mysql.connector import pooling
from flask import Flask, render_template, jsonify, request, url_for, Response, session, redirect
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from datetime import datetime
import threading
import time
import numpy as np
import base64
import smtplib
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from flask_talisman import Talisman
from flask_caching import Cache
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION ---
app.secret_key = os.environ.get('SECRET_KEY', 'AgriGuard_Cloud_Key_2026')
# Talisman helps with security, but we disable strict CSP for the video/canvas to work easily
Talisman(app, content_security_policy=None, force_https=False)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
cache = Cache(app)

DETECTIONS_FOLDER = os.path.join('static', 'detections')
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

# --- EMAIL CREDENTIALS ---
SMTP_EMAIL = os.environ.get("EMAIL_USER")
SMTP_PASSWORD = os.environ.get("EMAIL_PASS")

# --- GLOBAL VARS ---
system_status = {'status': 'OFF'}
model = None
otp_storage = {}
TARGET_ANIMALS = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'monkey', 'boar']

# --- MYSQL CONFIGURATION ---
db_config = {
    'host': os.environ.get('DB_HOST'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD'),
    'database': os.environ.get('DB_NAME'),
    'port': int(os.environ.get('DB_PORT', 3306))
}

connection_pool = None
try:
    if db_config['host']:
        connection_pool = pooling.MySQLConnectionPool(pool_name="agri_pool", pool_size=5, pool_reset_session=True, **db_config)
        print("✅ Cloud Database Pool Created.")
except Exception as e: 
    print(f"❌ DB Error: {e}")

# --- AI MODEL LOADING ---
def load_yolo():
    global model
    try:
        model = YOLO('models/yolo11n.pt') 
        print("✅ YOLOv11n Loaded successfully.")
    except Exception as e:
        print(f"❌ Model Loading Error: {e}")

# --- DATABASE LOGGING ---
def log_detection(animal_name):
    if connection_pool:
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor()
            now = datetime.now()
            # We don't save a physical image path here to save Render disk space, 
            # but we log the event.
            cursor.execute("INSERT INTO detections (timestamp, animal_name, image_path) VALUES (%s, %s, %s)", 
                           (now, animal_name, "Cloud_Detection"))
            conn.commit()
            cursor.close()
            conn.close()
            print(f"💾 Logged {animal_name} to database.")
        except Exception as e:
            print(f"❌ DB Log Error: {e}")

# --- ROUTES ---

@app.route('/')
def home():
    is_logged_in = 'user_id' in session
    return render_template('home.html', logged_in=is_logged_in)

@app.route('/live')
def live():
    if 'user_id' not in session: return redirect(url_for('home'))
    return render_template('live.html')

# --- THE NEW BROWSER-BASED DETECTION ROUTE ---
@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global system_status
    if system_status['status'] == 'OFF':
        return jsonify({"status": "system_off"})

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data"}), 400

    # 1. Decode Base64 image from browser
    try:
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Run AI Inference
        results = model.predict(source=frame, conf=0.5, imgsz=320, verbose=False)
        
        detected_list = []
        alert = False

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in TARGET_ANIMALS:
                    detected_list.append(label)
                    alert = True
                    # Log detection to MySQL
                    log_detection(label)

        return jsonify({
            "status": "success",
            "detected": detected_list,
            "alert": alert
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status/toggle', methods=['POST'])
def toggle_status():
    global system_status
    data = request.json
    system_status['status'] = data.get('status', 'OFF')
    return jsonify({'success': True, 'new_status': system_status['status']})

# --- (Keep your Registration/Login/OTP routes exactly as they were) ---

if __name__ == '__main__':
    load_yolo()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
