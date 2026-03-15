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
# Use the random string we generated earlier for your SECRET_KEY in Render
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_if_env_missing')
Talisman(app, content_security_policy=None, force_https=False)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
cache = Cache(app)

# --- EMAIL CREDENTIALS ---
SMTP_EMAIL = os.environ.get("EMAIL_USER")
SMTP_PASSWORD = os.environ.get("EMAIL_PASS")

# --- GLOBAL VARS ---
system_status = {'status': 'OFF'}
model = None
otp_storage = {}
TARGET_ANIMALS = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'monkey', 'boar']

# --- MYSQL CONFIGURATION ---
connection_pool = None
def init_connection_pool():
    global connection_pool
    try:
        host = os.environ.get('DB_HOST')
        if host:
            connection_pool = pooling.MySQLConnectionPool(
                pool_name="agri_pool",
                pool_size=5,
                pool_reset_session=True,
                host=host,
                user=os.environ.get('DB_USER'),
                password=os.environ.get('DB_PASSWORD'),
                database=os.environ.get('DB_NAME', 'agriguard_db'),
                port=int(os.environ.get('DB_PORT', 3306))
            )
            print("✅ Database Pool Created.")
    except Exception as e:
        print(f"❌ DB Pool Error: {e}")

# --- AI MODEL LOADING ---
def load_yolo():
    global model
    try:
        # Load model from the 'models' folder in your GitHub repo
        model = YOLO('models/yolo11n.pt') 
        print("✅ YOLOv11n Loaded.")
    except Exception as e:
        print(f"❌ Model Error: {e}")

# --- DATABASE HELPERS ---
def log_detection(animal_name):
    if not connection_pool: return
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO detections (timestamp, animal_name, image_path) VALUES (%s, %s, %s)", 
                       (datetime.now(), animal_name, "Cloud_Detection"))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"💾 DB Log Error: {e}")

def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"📧 Email Error: {e}")
        return False

# --- ROUTES ---

@app.route('/')
def home():
    is_logged_in = 'user_id' in session
    return render_template('home.html', logged_in=is_logged_in)

@app.route('/live')
def live():
    if 'user_id' not in session: 
        return redirect(url_for('home'))
    return render_template('live.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    if system_status['status'] == 'OFF':
        return jsonify({"status": "system_off"})

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No data"}), 400

    try:
        # Decode image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Inference
        results = model.predict(source=frame, conf=0.5, imgsz=320, verbose=False)
        detected = []
        alert = False

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in TARGET_ANIMALS:
                    detected.append(label)
                    alert = True
                    log_detection(label)

        return jsonify({"status": "success", "detected": detected, "alert": alert})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- AUTH ROUTES (Fixed & Added back) ---

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    # Placeholder for actual DB check - replace with your query logic
    # If using DB: conn = connection_pool.get_connection()...
    otp = str(random.randint(100000, 999999))
    otp_storage[email] = {'otp': otp, 'timestamp': time.time()}
    
    if send_email(email, "AgriGuard Login", f"Your OTP is: {otp}"):
        return jsonify({"success": True, "message": "OTP Sent"})
    return jsonify({"success": False, "message": "Email Error"}), 500

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    data = request.json
    email = data.get('email')
    user_otp = data.get('otp')
    
    if email in otp_storage and otp_storage[email]['otp'] == user_otp:
        session['user_id'] = email # Store something in session
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Invalid OTP"})

@app.route('/api/status/toggle', methods=['POST'])
def toggle_status():
    data = request.json
    system_status['status'] = data.get('status', 'OFF')
    return jsonify({'success': True, 'new_status': system_status['status']})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    init_connection_pool()
    load_yolo()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
