from flask import request, jsonify
import jwt
import datetime
from app import app
from app.database import conn, cursor

SECRET_KEY = "your_secret_key"

# Generate JWT Token
def generate_token(username):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


# Authentication Routes
@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data["username"]
    password = data["password"]

    cursor.execute("SELECT username FROM login_credentials WHERE username = %s AND hashed_password = %s",
                   (username, password))
    user = cursor.fetchone()

    if user:
        token = generate_token(user[0])
        return jsonify({"message": "Login successful", "token": token})
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    cursor.execute("INSERT INTO login_credentials (user_id, username, hashed_password, role) VALUES (gen_random_uuid(), %s, %s, %s)",
                   (data["username"], data["password"], data["role"]))
    conn.commit()
    return jsonify({"message": "User registered successfully"})

# Student Routes
@app.route("/students", methods=["POST"])
def add_student():
    data = request.get_json()
    cursor.execute("INSERT INTO students (student_id, name, department, level, face_embedding) VALUES (%s, %s, %s, %s, %s)",
                   (data["student_id"], data["name"], data["department"], data["level"], data["face_embedding"]))
    conn.commit()
    return jsonify({"message": "Student added successfully"})

@app.route("/students/<student_id>", methods=["GET"])
def get_student(student_id):
    cursor.execute("SELECT * FROM students WHERE student_id = %s", (student_id,))
    student = cursor.fetchone()
    return jsonify({"student": student})

# Lecturer Routes
@app.route("/lecturers", methods=["POST"])
def add_lecturer():
    data = request.get_json()
    cursor.execute("INSERT INTO lecturers (lecturer_id, name) VALUES (gen_random_uuid(), %s)", (data["name"],))
    conn.commit()
    return jsonify({"message": "Lecturer added successfully"})

@app.route("/lecturers/<lecturer_id>", methods=["GET"])
def get_lecturer(lecturer_id):
    cursor.execute("SELECT * FROM lecturers WHERE lecturer_id = %s", (lecturer_id,))
    lecturer = cursor.fetchone()
    return jsonify({"lecturer": lecturer})

# Course Routes
@app.route("/courses", methods=["POST"])
def add_course():
    data = request.get_json()
    cursor.execute("INSERT INTO course (course_code, course_name, lecturer_id) VALUES (%s, %s, %s)",
                   (data["course_code"], data["course_name"], data["lecturer_id"]))
    conn.commit()
    return jsonify({"message": "Course added successfully"})

@app.route("/courses/<course_code>", methods=["GET"])
def get_course(course_code):
    cursor.execute("SELECT * FROM course WHERE course_code = %s", (course_code,))
    course = cursor.fetchone()
    return jsonify({"course": course})

# Enrollment Routes
@app.route("/enrollment", methods=["POST"])
def enroll_student():
    data = request.get_json()
    cursor.execute("INSERT INTO enrollment (enrollment_id, student_id, course_code, enrollment_date) VALUES (gen_random_uuid(), %s, %s, CURRENT_TIMESTAMP)",
                   (data["student_id"], data["course_code"]))
    conn.commit()
    return jsonify({"message": "Student enrolled successfully"})

@app.route("/enrollment/<student_id>", methods=["GET"])
def get_student_enrollment(student_id):
    cursor.execute("SELECT * FROM enrollment WHERE student_id = %s", (student_id,))
    enrollment = cursor.fetchall()
    return jsonify({"enrollment": enrollment})

# Attendance Routes
@app.route("/attendance/recognize", methods=["POST"])
def mark_attendance():
    data = request.get_json()
    student_id = data["student_id"]
    course_code = data["course_code"]

    cursor.execute("INSERT INTO attendance_record (student_id, course_code, date, check_in_time, status) VALUES (%s, %s, CURRENT_DATE, CURRENT_TIMESTAMP, 'Present')",
                   (student_id, course_code))
    conn.commit()

    return jsonify({"message": "Attendance recorded"})

@app.route("/attendance/<student_id>", methods=["GET"])
def get_attendance(student_id):
    cursor.execute("SELECT * FROM attendance_record WHERE student_id = %s", (student_id,))
    attendance = cursor.fetchall()
    return jsonify({"attendance": attendance})
