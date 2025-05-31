from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import datetime
import os
from dotenv import load_dotenv
from groq import Groq
import json
import base64
from typing import Dict, Any, List

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
if not firebase_admin._apps and firebase_creds:
    cred = credentials.Certificate(json.loads(firebase_creds))
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://elderlycare-60475-default-rtdb.firebaseio.com/'
    })

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global conversation history for users without accounts
global_conversation_history = []

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

def get_conversation_history(user_name: str, max_messages: int = 5) -> List[Dict[str, str]]:
    """Get recent conversation history for a user"""
    history_ref = db.reference(f"conversations/{user_name}")
    history = history_ref.order_by_key().limit_to_last(max_messages).get()
    
    # Convert to list of messages in chronological order
    if history:
        messages = []
        for timestamp, data in sorted(history.items()):
            messages.append({
                "role": data.get("role", "user"),
                "content": data.get("content", "")
            })
        return messages
    return []

def save_conversation_entry(user_name: str, role: str, content: str) -> None:
    """Save a new conversation entry to the database"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    history_ref = db.reference(f"conversations/{user_name}/{timestamp}")
    history_ref.set({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })

@app.route("/chat", methods=['POST'])
def unified_chat():
    """Chat endpoint that works without requiring user_name"""
    global global_conversation_history  # Add this line to access the global variable
    
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    message = data['message']
    
    try:
        # Build the system prompt
        system_prompt = {
            "role": "system",
            "content": (
                "You are GoldenCare, an advanced medical assistant developed by NirveonX. "
                "You have access to the user's health profile. "
                "Provide personalized, evidence-based advice while maintaining a warm and caring tone. "
                "include relevant emojis, and end with a helpful follow-up question. "
                "Some time provide direct diagnosis but guide users toward healthy habits and professional medical consultation when needed. "
                "Reply in context of the previous chat history, medical history, and daily check-in data. "
                "Use the user's name occasionally, but not in every message to maintain a natural conversation"
            )
        }
        
        # Prepare messages for the API call
        messages = [system_prompt]
        
        # Add conversation history
        if global_conversation_history:
            # Only use the last 5 messages to avoid context overflow
            messages.extend(global_conversation_history[-5:])
        
        # Add the current user message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            max_tokens=250,
            temperature=0.6,
            top_p=0.9,
        )
        
        assistant_response = response.choices[0].message.content
        
        # Save the conversation entries to global history
        global_conversation_history.append({"role": "user", "content": message})
        global_conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Limit history size to avoid memory issues
        if len(global_conversation_history) > 20:
            global_conversation_history = global_conversation_history[-20:]
        
        return jsonify({"response": assistant_response})
    
    except Exception as e:
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

@app.route("/clear-chat", methods=['POST'])
def clear_chat():
    """Clear the current chat history"""
    global global_conversation_history  # Add this line to access the global variable
    global_conversation_history = []
    return jsonify({"status": "success", "message": "Chat history cleared"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)