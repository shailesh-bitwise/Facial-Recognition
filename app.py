# Updated Flask App Code (app.py)

import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, session, g, send_file, jsonify
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import sqlite3

# Defining Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

DATABASE = 'attendance.db'

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# SQLite3 Database Connection
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# If these directories don't exist, create them
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# get a number of total registered users
def totalreg():
    db = get_db()
    cur = db.execute('SELECT COUNT(*) FROM users')
    count = cur.fetchone()[0]
    return count

# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    db = get_db()
    cur = db.execute('SELECT username, userid, timestamp FROM attendance WHERE DATE(timestamp) = DATE("now")')
    rows = cur.fetchall()
    names = [row[0] for row in rows]
    rolls = [row[1] for row in rows]
    times = [row[2] for row in rows]
    l = len(rows)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db = get_db()
    db.execute('INSERT INTO attendance (username, userid, timestamp) VALUES (?, ?, ?)', (username, userid, current_time))
    db.commit()

# A function to get names and rol numbers of all users
def getallusers():
    db = get_db()
    cur = db.execute('SELECT username, userid, status FROM users')
    rows = cur.fetchall()
    names = [row[0] for row in rows]
    rolls = [row[1] for row in rows]
    statuses = [row[2] for row in rows]
    l = len(rows)
    return rows, names, rolls, statuses, l

# A function to delete a user from the database and their folder
def delete_user(userid):
    db = get_db()
    cur = db.execute('SELECT username FROM users WHERE userid = ?', (userid,))
    row = cur.fetchone()
    if row:
        username = row[0]
        userfolder = f'static/faces/{username}_{userid}'
        if os.path.exists(userfolder):
            for img in os.listdir(userfolder):
                os.remove(os.path.join(userfolder, img))
            os.rmdir(userfolder)
        db.execute('DELETE FROM users WHERE userid = ?', (userid,))
        db.commit()

# Extract specific user's attendance
def get_user_attendance(username):
    db = get_db()
    cur = db.execute('SELECT username, userid, timestamp FROM attendance WHERE username = ?', (username,))
    rows = cur.fetchall()
    names = [row[0] for row in rows]
    rolls = [row[1] for row in rows]
    times = [row[2] for row in rows]
    l = len(rows)
    return names, rolls, times, l

################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    return render_template('home.html', datetoday2=datetoday2)

# Admin login page
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Replace with your admin credentials
        if username == 'admin123@msk.com' and password == 'msksolutions':
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
    return render_template('admin_login.html')

# Admin logout
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('home'))

# Admin dashboard
@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    names, rolls, times, l = extract_attendance()
    userlist, usernames, userrolls, statuses, usercount = getallusers()
    return render_template('admin_dashboard.html', names=names, rolls=rolls, times=times, l=l, userlist=userlist, usernames=usernames, userrolls=userrolls, statuses=statuses, usercount=usercount, totalreg=totalreg(), datetoday2=datetoday2)

# List users page
@app.route('/admin/listusers')
def listusers():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    userlist, names, rolls, statuses, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Delete functionality
@app.route('/admin/deleteuser', methods=['GET'])
def deleteuser():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    userid = request.args.get('userid')
    delete_user(userid)
    userlist, names, rolls, statuses, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, statuses=statuses, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/admin/updateuserstatus', methods=['POST'])
def update_user_status():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    userid = request.form['userid']
    status = request.form['status']
    db = get_db()
    db.execute('UPDATE users SET status = ? WHERE userid = ?', (status, userid))
    db.commit()
    return redirect(url_for('admin_dashboard'))

# User signup page
@app.route('/user/signup', methods=['GET', 'POST'])
def user_signup():
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        db = get_db()
        cur = db.execute('SELECT * FROM users WHERE userid = ?', (newuserid,))
        existing_user = cur.fetchone()

        if existing_user:
            return render_template('user_signup.html', message="User ID already exists. Please choose a different one.")

        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            if j == nimgs * 5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        db.execute('INSERT INTO users (username, userid, status) VALUES (?, ?, "accept")', (newusername, newuserid))
        db.commit()
        train_model()
        return redirect(url_for('user_login'))
    return render_template('user_signup.html')

# User login page
@app.route('/user', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        username = request.form['username']
        userid = request.form['userid']
        db = get_db()
        cur = db.execute('SELECT username, status FROM users WHERE username = ? AND userid = ?', (username, userid))
        row = cur.fetchone()
        if row:
            if row[1] == "accept":
                session['user'] = username
                return redirect(url_for('user_dashboard'))
            elif row[1] == "deny":
                return render_template('user_login.html', message="Permission denied")
            else:
                return render_template('user_login.html', message="Your account is pending approval")
        else:
            return render_template('user_login.html', message="User not found or ID incorrect")
    return render_template('user_login.html')

@app.route('/user/logout')
def user_logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/user/dashboard')
def user_dashboard():
    if 'user' not in session:
        return redirect(url_for('user_login'))
    username = session['user']
    names, rolls, times, l = get_user_attendance(username)
    return render_template('user_dashboard.html', username=username, names=names, rolls=rolls, times=times, l=l, datetoday2=datetoday2)

@app.route('/user/edit', methods=['GET', 'POST'])
def user_edit():
    if 'user' not in session:
        return redirect(url_for('user_login'))
    username = session['user']
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        db = get_db()
        db.execute('UPDATE users SET username = ?, userid = ? WHERE username = ?', (newusername, newuserid, username))
        db.commit()
        session['user'] = newusername
        return redirect(url_for('user_dashboard'))
    return render_template('user_edit.html', username=username)

# Remove user from the database
@app.route('/admin/removeuser', methods=['POST'])
def remove_user():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    userid = request.form['userid']
    delete_user(userid)
    return redirect(url_for('admin_dashboard'))

# Our main Face Recognition functionality.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret = False  # Break the loop after attendance is taken
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1):
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Export attendance data
@app.route('/export', methods=['GET'])
def export():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    format_type = request.args.get('format')
    db = get_db()
    cur = db.execute('SELECT * FROM attendance WHERE DATE(timestamp) = DATE("now")')
    rows = cur.fetchall()

    if format_type == 'csv':
        with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
            f.write('Name,Roll,Time\n')
            for row in rows:
                f.write(f'{row[1]},{row[2]},{row[3]}\n')
        return send_file(f'Attendance/Attendance-{datetoday}.csv', as_attachment=True,
                         download_name=f'Attendance-{datetoday}.csv')

    elif format_type == 'excel':
        df = pd.DataFrame(rows, columns=["ID", "Name", "Roll", "Time"])
        excel_file_path = f'Attendance/Attendance-{datetoday}.xlsx'
        df.to_excel(excel_file_path, index=False)
        return send_file(excel_file_path, as_attachment=True, download_name=f'Attendance-{datetoday}.xlsx')

    elif format_type == 'json':
        attendance_list = [{"Name": row[1], "Roll": row[2], "Time": row[3]} for row in rows]
        return jsonify(attendance_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0')