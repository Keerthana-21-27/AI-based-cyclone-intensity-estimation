import os
import cv2
import re
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
IMG_SIZE = (128, 128)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------
# Load Model
# -----------------------
model = load_model("Models/resnet50_model.h5", compile=False)

# -----------------------
# Utility Functions
# -----------------------
def load_and_preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = preprocess_input(img_resized.astype(np.float32))
    return img, np.expand_dims(img_array, axis=0)

def save_bt_contour(img, out_path):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.figure()
    plt.contour(gray, levels=10, cmap='jet')
    plt.colorbar(label="Brightness Temperature")
    plt.savefig(out_path)
    plt.close()

def save_bt_calibrated(img, out_path):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bt = 200 + (gray / 255.0) * 100  # pseudo BT calibration
    plt.imshow(bt, cmap="jet")
    plt.colorbar(label="BT (K)")
    plt.axis("off")
    plt.savefig(out_path)
    plt.close()

def save_cloud_height_3d(img, out_path):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height = (255 - gray) / 255.0 * 15  # km

    X, Y = np.meshgrid(
        np.arange(height.shape[1]),
        np.arange(height.shape[0])
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, height, cmap="gray")
    ax.set_zlabel("Cloud Height (km)")
    plt.savefig(out_path)
    plt.close()

# -----------------------
# Routes
# -----------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            raw_img, model_input = load_and_preprocess(img_path)
            prediction = model.predict(model_input)[0][0]

            # Output paths
            raw_out = os.path.join(OUTPUT_FOLDER, "raw.png")
            bt_contour = os.path.join(OUTPUT_FOLDER, "bt_contour.png")
            bt_cal = os.path.join(OUTPUT_FOLDER, "bt_cal.png")
            cloud_3d = os.path.join(OUTPUT_FOLDER, "cloud_3d.png")

            # Save raw
            cv2.imwrite(raw_out, cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))

            save_bt_contour(raw_img, bt_contour)
            save_bt_calibrated(raw_img, bt_cal)
            save_cloud_height_3d(raw_img, cloud_3d)

            return render_template(
                "result.html",
                prediction=round(float(prediction), 2),
                raw="raw.png",
                bt_contour="bt_contour.png",
                bt_cal="bt_cal.png",
                cloud_3d="cloud_3d.png"
            )

    return render_template("home.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route("/graphs")
def graphs():
    return render_template("graphs.html")

@app.route("/about2")
def about2():
    return render_template("about2.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
