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
import requests
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
# Use Render Environment Variable for the secret key
app.secret_key = os.environ.get('SECRET_KEY', 'AgriGuard_Cloud_Key_2026')
Talisman(app, content_security_policy=None, force_https=False)
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
cache = Cache(app)

DETECTIONS_FOLDER = os.path.join('static', 'detections')
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

# --- EMAIL CREDENTIALS (FROM ENV) ---
SMTP_EMAIL = os.environ.get("EMAIL_USER")
SMTP_PASSWORD = os.environ.get("EMAIL_PASS")

# --- GLOBAL VARS ---
system_status = {'status': 'OFF'}
current_frame = None
lock = threading.Lock()
last_detection_time = None
model = None
otp_storage = {}

# --- MYSQL CONFIGURATION (UPDATED FOR CLOUD) ---
db_config = {
    'host': os.environ.get('DB_HOST'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD'),
    'database': os.environ.get('DB_NAME'),
    'port': int(os.environ.get('DB_PORT', 3306))
}

connection_pool = None
try:
    if db_config['host']: # Only attempt if host is provided
        connection_pool = pooling.MySQLConnectionPool(pool_name="agri_pool", pool_size=5, pool_reset_session=True, **db_config)
        print("✅ Cloud Database Pool Created.")
except Exception as e: 
    print(f"❌ DB Error: {e}")

# --- HELPER: SEND EMAIL ---
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
        print(f"Email Error: {e}")
        return False

# --- ALERT LOGIC (REMOVED PLAYSOUND) ---
def trigger_remote_alert(animal_name):
    # playsound is removed because Render has no speakers.
    # Instead, we log it and you can use Frontend JS to play sound.
    print(f"🚨 CLOUD ALERT: {animal_name} detected!")

# --- SURVEILLANCE LOOP (MODIFIED FOR RENDER) ---
def surveillance_loop():
    global current_frame, system_status, last_detection_time, model
    
    TARGET_ANIMALS = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'monkey', 'boar']

    if model is None:
        try:
            # Use relative path for Render
            model = YOLO('models/yolo11n.pt') 
            print("✅ AI Model Loaded.")
        except Exception as e: 
            print(f"❌ Model Error: {e}")
            model = None
    
    # RENDER NOTE: Render cannot access your laptop camera (cv2.VideoCapture(0)).
    # This loop will now act as a placeholder. In production, you would
    # POST frames from your phone/camera to a specific route.
    
    while True:
        if system_status['status'] == 'OFF':
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "SYSTEM OFFLINE", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            with lock: current_frame = blank.copy()
            time.sleep(1)
            continue
        
        # Simulating frame processing or waiting for upload
        time.sleep(1)

# --- DATABASE INIT ---
def init_db():
    if connection_pool is None: return
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, first_name VARCHAR(255) NOT NULL, last_name VARCHAR(255) NOT NULL, email VARCHAR(255) NOT NULL UNIQUE, password_hash VARCHAR(255) NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS detections (id INT AUTO_INCREMENT PRIMARY KEY, timestamp DATETIME NOT NULL, animal_name VARCHAR(255) NOT NULL, image_path VARCHAR(255) NOT NULL)''')
        conn.commit(); cursor.close(); conn.close()
    except Exception as e: print(f"DB Init Error: {e}")

# ... (Keep your existing AUTH and API routes here, they work fine!) ...

if __name__ == '__main__':
    init_db()
    t = threading.Thread(target=surveillance_loop)
    t.daemon = True
    t.start()
    
    # IMPORTANT: Render requires the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
