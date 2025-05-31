from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import datetime
import os
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json
import base64
from werkzeug.utils import secure_filename
from typing import Optional, Dict, Any, List

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize Firebase
firebase_creds_b64 = os.getenv("FIREBASE_CREDENTIALS")
firebase_creds = base64.b64decode(firebase_creds_b64).decode("utf-8")
# print("Firebase credentials:", firebase_creds)
if not firebase_admin._apps and firebase_creds:
    cred = credentials.Certificate(json.loads(firebase_creds))
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://elderlycare-60475-default-rtdb.firebaseio.com/'
    })

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

@app.route("/", methods=['GET'])
def root():
    return jsonify({"message": "Welcome to Elderly Care API"})

@app.route("/users/<user_name>", methods=['GET'])
def get_user(user_name):
    """Get user profile data if it exists"""
    user_ref = db.reference(f"users/{user_name}")
    user_data = user_ref.get()
    if not user_data:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user_data)

@app.route("/users", methods=['POST'])
def create_user():
    """Create a new user health profile"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Validate required fields
    if 'name' not in data:
        return jsonify({"error": "Name is required"}), 400
    
    user_ref = db.reference(f"users/{data['name']}")
    existing_user = user_ref.get()
    if existing_user:
        return jsonify({"error": "User already exists"}), 400
    
    # Set default values if not provided
    health_profile = {
        "name": data['name'],
        "age": data.get('age', 0),
        "conditions": data.get('conditions', []),
        "bp_baseline": data.get('bp_baseline', {"systolic": 0, "diastolic": 0}),
        "sugar_baseline": data.get('sugar_baseline', 0),
        "lifestyle": data.get('lifestyle', {
            "diet": "Balanced",
            "exercise": "None",
            "medications": []
        })
    }
    
    user_ref.set(health_profile)
    return jsonify({"message": "Health profile created successfully", "profile": health_profile})

@app.route("/users/<user_name>/checkin", methods=['POST'])
def daily_checkin(user_name):
    """Submit a daily health check-in"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    user_ref = db.reference(f"users/{user_name}")
    user_data = user_ref.get()
    if not user_data:
        return jsonify({"error": "User not found"}), 404
    
    today = str(datetime.date.today())
    checkin_data = {
        "date": today,
        "bp": data.get('bp', {"systolic": 0, "diastolic": 0}),
        "sugar": data.get('sugar', 0),
        "symptoms": data.get('symptoms', []),
        "mood": data.get('mood', 3),
        "medication_taken": data.get('medication_taken', True),
        "sleep": data.get('sleep', "Good"),
        "appetite": data.get('appetite', "Normal")
    }
    
    user_ref.child("daily_logs").child(today).set(checkin_data)
    return jsonify({"message": "Check-in submitted successfully", "date": today})

@app.route("/users/<user_name>/analyze", methods=['POST'])
def analyze_health(user_name):
    """Analyze user's health data and provide advice"""
    user_ref = db.reference(f"users/{user_name}")
    user_data = user_ref.get()
    if not user_data:
        return jsonify({"error": "User not found"}), 404
    
    today = str(datetime.date.today())
    daily_log = user_ref.child("daily_logs").child(today).get()
    
    if not daily_log:
        return jsonify({"error": "No check-in data found for today"}), 404
    
    try:
        prompt = f"""
        Act as a compassionate elderly health advisor. Here's a daily health update:
        Name: {user_name}
        Age: {user_data.get('age')}
        Conditions: {', '.join(user_data.get('conditions', []))}
        Today's BP: {daily_log.get('bp', {}).get('systolic')}/{daily_log.get('bp', {}).get('diastolic')}
        Sugar: {daily_log.get('sugar')}
        Symptoms: {', '.join(daily_log.get('symptoms', []))}
        Mood (1-5): {daily_log.get('mood')}
        Sleep: {daily_log.get('sleep')}
        Appetite: {daily_log.get('appetite')}
        Medications Taken: {daily_log.get('medication_taken')}

        Give brief health advice tailored to this update. Avoid generic responses. Keep it around 100 words.
        """
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a senior health assistant offering brief, clear, and friendly advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.6,
        )
        
        advice = response.choices[0].message.content
        return jsonify({"advice": advice})
    
    except Exception as e:
        return jsonify({"error": f"Failed to generate advice: {str(e)}"}), 500

def get_user_context(user_name: str) -> Dict[str, Any]:
    """Get user's health profile and recent check-ins"""
    user_ref = db.reference(f"users/{user_name}")
    user_data = user_ref.get()
    if not user_data:
        return None
    
    # Get today's check-in if available
    today = str(datetime.date.today())
    daily_log = user_ref.child("daily_logs").child(today).get()
    
    return {
        "profile": user_data,
        "daily_log": daily_log
    }

def get_medical_report_context(message: str) -> Optional[str]:
    """Get relevant context from medical report if available"""
    if not vector_store:
        return None
    
    try:
        docs = vector_store.similarity_search(message, k=3)
        return "\n".join([doc.page_content for doc in docs])
    except Exception:
        return None

def should_use_medical_context(message: str) -> bool:
    """Determine if the query is related to medical reports"""
    medical_keywords = [
        "report", "test", "result", "diagnosis", "scan", "x-ray", "mri", "ct",
        "blood", "lab", "medical", "doctor", "hospital", "treatment", "prescription",
        "medication", "symptom", "condition", "disease", "illness"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in medical_keywords)

@app.route("/chat", methods=['POST'])
def unified_chat():
    """Unified chat endpoint that intelligently handles both general queries and medical report analysis"""
    data = request.get_json()
    if not data or 'message' not in data or 'user_name' not in data:
        return jsonify({"error": "Message and user_name are required"}), 400
    
    message = data['message']
    user_name = data['user_name']
    
    try:
        # Get user context
        user_context = get_user_context(user_name)
        if not user_context:
            return jsonify({"error": "User not found"}), 404
        
        # Determine if we should use medical report context
        use_medical_context = should_use_medical_context(message)
        medical_context = get_medical_report_context(message) if use_medical_context else None
        
        # Build the system prompt
        system_prompt = {
            "role": "system",
            "content": (
                "You are GoldenCare, an advanced medical assistant developed by NirveonX. "
                "You have access to the user's health profile and medical reports when available. "
                "Provide personalized, evidence-based advice while maintaining a warm and caring tone. "
                "Keep responses under 150 words, include relevant emojis, and end with a helpful follow-up question. "
                "Never provide direct diagnosis but guide users toward healthy habits and professional medical consultation when needed. "
                "When discussing medical reports, focus on explaining the information in simple terms and suggesting next steps."
            )
        }
        
        # Build the user prompt with all available context
        profile_section = (
            f"User Profile:\n"
            f"Name: {user_name}\n"
            f"Age: {user_context['profile'].get('age')}\n"
            f"Conditions: {', '.join(user_context['profile'].get('conditions', []))}"
        )
        
        health_status = (
            f"Today's Health Status:\n"
            f"{json.dumps(user_context['daily_log'], indent=2) if user_context['daily_log'] else 'No check-in data for today'}"
        )
        
        medical_section = f"Relevant Medical Report Context:\n{medical_context}" if medical_context else ""
        
        user_prompt = f"{profile_section}\n\n{health_status}\n\n{medical_section}\n\nUser Question: {message}\n\nPlease provide a personalized response based on the available information."
        
        # Generate response
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                system_prompt,
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=250,
            temperature=0.6,
            top_p=0.9,
        )
        
        # Prepare response with sources if medical report context was used
        response_data = {
            "response": response.choices[0].message.content,
            "confidence": 0.9 if medical_context else 1.0
        }
        
        if medical_context:
            response_data["sources"] = ["Medical Report Analysis"]
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

def process_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        # Read PDF content
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        return text
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def create_vector_store(text: str):
    """Create vector store from text"""
    global vector_store
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload-medical-report", methods=['POST'])
def upload_medical_report():
    """Upload and process medical report PDF"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    user_name = request.form.get('user_name')
    if not user_name:
        return jsonify({"error": "user_name is required"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400
    
    try:
        # Process PDF
        text = process_pdf(file)
        
        # Create vector store
        create_vector_store(text)
        
        # Store reference in Firebase
        user_ref = db.reference(f"users/{user_name}")
        user_data = user_ref.get()
        if not user_data:
            return jsonify({"error": "User not found"}), 404
        
        # Add report to user's medical reports
        report_data = {
            "filename": file.filename,
            "upload_date": str(datetime.datetime.now()),
            "processed": True
        }
        
        user_ref.child("medical_reports").push(report_data)
        
        return jsonify({"message": "Medical report processed successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat-with-report", methods=['POST'])
def chat_with_report():
    """Chat with the medical report using vector store"""
    data = request.get_json()
    if not data or 'message' not in data or 'user_name' not in data:
        return jsonify({"error": "Message and user_name are required"}), 400
    
    if not vector_store:
        return jsonify({"error": "No medical report has been processed yet"}), 400
    
    try:
        # Get user data from Firebase
        user_ref = db.reference(f"users/{data['user_name']}")
        user_data = user_ref.get()
        if not user_data:
            return jsonify({"error": "User not found"}), 404
        
        # Search relevant context from vector store
        docs = vector_store.similarity_search(data['message'], k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Create prompt with user data and report context
        prompt = f"""
        Act as a compassionate medical assistant. Here's the context from the medical report and user profile:
        
        User Profile:
        Name: {data['user_name']}
        Age: {user_data.get('age')}
        Conditions: {', '.join(user_data.get('conditions', []))}
        
        Relevant Report Context:
        {context}
        
        User Question: {data['message']}
        
        Provide a personalized response based on the medical report and user profile. Keep it under 150 words.
        """
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a medical assistant offering personalized advice based on medical reports and user profiles."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.6,
        )
        
        return jsonify({"response": response.choices[0].message.content})
    
    except Exception as e:
        return jsonify({"error": f"Error processing chat request: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)