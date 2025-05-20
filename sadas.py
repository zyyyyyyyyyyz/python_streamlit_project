import streamlit as st
import pandas as pd
from pandas.tseries.offsets import DateOffset
import json
import sys
import time
from datetime import datetime
import plotly.express as px
from streamlit_lottie import st_lottie
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import requests
import warnings
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client

# Streamlit page configuration
st.set_page_config(
    page_title="AI Healthcare Automation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

# Function to make Hugging Face API call
def call_huggingface_api(prompt, max_length=800):
    data = {
        "inputs": prompt,
        "parameters": {
            "max_length": max_length,
            "temperature": 0.3,
            "return_full_text": False
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            raw_response = result[0]["generated_text"]
            return raw_response
        else:
            st.error(f"Unexpected response format: {result}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Hugging Face API call failed: {str(e)}")
        return None

# Function to validate analytics dictionary
def validate_analytics(analytics):
    required_keys = ["triage_stats", "medication_stats", "mental_health_stats", "report_stats"]
    return all(key in analytics for key in required_keys)

# Function to normalize and validate triage output
def normalize_triage_output(output):
    required_keys = [
        "symptoms", "duration", "medical_history", "urgency",
        "triage_category", "recommended_tests", "potential_diagnosis", "action"
    ]
    for key in required_keys:
        if key not in output:
            if key in ["symptoms", "medical_history", "recommended_tests", "potential_diagnosis"]:
                output[key] = []
            elif key == "action":
                output[key] = "No action specified"
            else:
                output[key] = "Unknown"
    if isinstance(output["symptoms"], str):
        output["symptoms"] = [output["symptoms"].strip()] if output["symptoms"].strip() else []
    return output

# Function to normalize medication adherence output
def normalize_medication_output(output):
    required_keys = [
        "medication", "dosage", "frequency", "timing", "duration",
        "patient_concern", "adherence_risk", "recommendation", "refill_date", "action"
    ]
    for key in required_keys:
        if key not in output or not output[key]:
            if key == "action":
                output[key] = "Patient education on migraine triggers scheduled + medication access reminder set" if output.get("medication") == "Sumatriptan" else "No action specified"
            elif key == "refill_date":
                output[key] = "N/A - as needed" if output.get("medication") == "Sumatriptan" else "N/A"
            elif key == "adherence_risk":
                output[key] = "Low" if output.get("medication") == "Sumatriptan" else "Unknown"
            elif key == "patient_concern":
                output[key] = "None reported"
            elif key == "recommendation":
                output[key] = "Keep medication accessible + migraine trigger tracking app" if output.get("medication") == "Sumatriptan" else "Not specified"
            elif key == "frequency" and output.get("medication") == "Sumatriptan":
                output[key] = "as needed"
            elif key == "frequency":
                output[key] = "Not specified"
            elif key == "timing" and output.get("medication") == "Sumatriptan":
                output[key] = "at onset of migraine"
            elif key == "duration" and output.get("medication") == "Sumatriptan":
                output[key] = "as needed for migraines"
            else:
                output[key] = "Not specified"
    return output

# Function to normalize mental health output
def normalize_mental_health_output(output):
    required_keys = [
        "risk_phrases", "symptoms", "risk_level", "suicide_risk",
        "recommended_response", "suggested_resources", "action"
    ]
    for key in required_keys:
        if key not in output:
            if key in ["risk_phrases", "symptoms", "suggested_resources"]:
                output[key] = []
            elif key == "action":
                output[key] = "No action specified"
            else:
                output[key] = "Unknown"
    return output

# Function to fetch analytics data using Hugging Face API
def get_analytics_from_gpt():
    prompt = """
You are an AI assistant that generates structured healthcare analytics data in JSON format. Generate analytics data for a healthcare dashboard with the following sections:
- Automated Triage System (total cases, urgency distribution, accuracy rate, avg processing time)
- Medication Adherence Assistant (adherence improvement, total reminders sent, most common medications, avg reminder response)
- Mental Health Crisis Detector (risk levels, intervention success rate, response time)
- Clinical Report Generator (total reports, avg completion time, error rate, physician satisfaction)
Ensure that 'most_common_medications' is a dictionary with medication names as keys and counts as values.
Return the response as a structured JSON object enclosed in triple backticks (```json\n...\n```). Example:
```json
{
  "triage_stats": {
    "total_cases": 600,
    "urgency_distribution": {"Critical": 90, "High": 160, "Medium": 200, "Low": 150},
    "accuracy_rate": 95.0,
    "avg_processing_time": 3.0
  },
  "medication_stats": {
    "adherence_improvement": 37,
    "total_reminders_sent": 12453,
    "most_common_medications": {"Metformin": 245, "Lisinopril": 198, "Atorvastatin": 176, "Levothyroxine": 145, "Amlodipine": 121},
    "avg_reminder_response": 85.7
  },
  "mental_health_stats": {...},
  "report_stats": {...}
}
```
"""
    response_text = call_huggingface_api(prompt, max_length=800)
    if response_text:
        try:
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:-3].strip()
            analytics = json.loads(json_str)
            if validate_analytics(analytics):
                return analytics
            else:
                st.error("Analytics data missing required keys. Using default analytics.")
                return default_analytics
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse analytics JSON: {str(e)}")
            return default_analytics
        except Exception as e:
            st.error(f"Unexpected error parsing analytics: {str(e)}")
            return default_analytics
    return default_analytics

# Function to load Lottie animations (using placeholder since network calls aren't allowed)
def load_lottieurl(url):
    return {"mock": "animation"}

# Load animations (mocked)
lottie_health = load_lottieurl("https://assets.lottiefiles.com/packages/lf20_jr9h1z0w.json")
lottie_medication = load_lottieurl("https://assets.lottiefiles.com/packages/lf20_m3juyggv.json")
lottie_mental = load_lottieurl("https://assets.lottiefiles.com/packages/lf20_ya1v7f61.json")
lottie_report = load_lottieurl("https://assets.lottiefiles.com/packages/lf20_kfl4ksf3.json")

# Custom CSS with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
    }
    
    .main .block-container {
        padding: 2rem;
    }
    
    h1, h2, h3 {
        color: #1E3A8A;
    }
    
    .stCard {
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        background: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1E3A8A, #3B82F6);
        color: white;
        padding: 20px;
        border-radius: 0;
        height: 100vh;
    }
    
    .sidebar-logo {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeIn 1s ease-in-out;
    }
    
    .sidebar-welcome {
        font-size: 14px;
        color: #E0F2FE;
        margin-bottom: 30px;
        text-align: center;
        animation: fadeInUp 1s ease-in-out 0.5s both;
    }
    
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        padding: 12px 15px;
        font-size: 14px;
        box-shadow: none !important;
        transition: border 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #3B82F6;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6, #1E3A8A);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(59, 130, 246, 0.3);
        background: linear-gradient(135deg, #2563EB, #1E40AF);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .animate-slide-in {
        animation: slideInLeft 0.5s ease-out forwards;
    }
    
    .loader {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    
    .loader-dot {
        background-color: #3B82F6;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin: 0 5px;
        animation: bounce 1.5s infinite ease-in-out;
    }
    
    .loader-dot:nth-child(1) { animation-delay: 0s; }
    .loader-dot:nth-child(2) { animation-delay: 0.2s; }
    .loader-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .status-critical {
        color: white;
        background-color: #EF4444;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-high {
        color: white;
        background-color: #F59E0B;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-medium {
        color: white;
        background-color: #3B82F6;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-low {
        color: white;
        background-color: #10B981;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .page-transition {
        animation: fadeIn 0.5s ease=in-out;
    }
    
    .json-output {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    
    .success-message {
        display: flex;
        align-items: center;
        background-color: #ECFDF5;
        color: #065F46;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 5px solid #10B981;
        margin-top: 15px;
        font-weight: 500;
    }
    
    .success-message i {
        margin-right: 10px;
        font-size: 18px;
    }
    
    .info-message {
        display: flex;
        align-items: center;
        background-color: #EFF6FF;
        color: #1E40AF;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
        margin-top: 15px;
        font-weight: 500;
    }
    
    .info-message i {
        margin-right: 10px;
        font-size: 18px;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        flex: 1;
        margin-right: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:last-child {
        margin-right: 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-title {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1E3A8A;
    }
    
    .metric-change {
        font-size: 12px;
        margin-top: 5px;
    }
    
    .metric-up {
        color: #10B981;
    }
    
    .metric-down {
        color: #EF4444;
    }
</style>
""", unsafe_allow_html=True)

# Default analytics data (fallback)
default_analytics = {
    "triage_stats": {
        "total_cases": 527,
        "urgency_distribution": {"Critical": 87, "High": 156, "Medium": 184, "Low": 100},
        "accuracy_rate": 94.3,
        "avg_processing_time": 3.2
    },
    "medication_stats": {
        "adherence_improvement": 37,
        "total_reminders_sent": 12453,
        "most_common_medications": {"Metformin": 245, "Lisinopril": 198, "Atorvastatin": 176, "Levothyroxine": 145, "Amlodipine": 121},
        "avg_reminder_response": 85.7
    },
    "mental_health_stats": {
        "risk_levels": {"High": 67, "Medium": 134, "Low": 289},
        "intervention_success_rate": 89.2,
        "response_time": 5.4
    },
    "report_stats": {
        "total_reports": 843,
        "avg_completion_time": 4.6,
        "error_rate": 1.2,
        "physician_satisfaction": 4.7
    }
}

# Fetch analytics data
analytics = get_analytics_from_gpt()
if not validate_analytics(analytics):
    st.error("Invalid analytics data structure. Using default analytics.")
    analytics = default_analytics

# Function to display loading animation
def loading_animation():
    with st.spinner('Processing...'):
        cols = st.columns([1, 1, 1, 1, 1])
        for i in range(3):
            for j, col in enumerate(cols[:3]):
                if i == 0:
                    with col:
                        st.markdown(f"<div class='loader-dot'></div>", unsafe_allow_html=True)
                    time.sleep(0.1)
                else:
                    time.sleep(0.1)

# Function to display a metric card
def metric_card(title, value, change=None, is_positive=True):
    change_html = ""
    if change is not None:
        icon = "↑" if is_positive else "↓"
        change_class = "metric-up" if is_positive else "metric-down"
        change_html = f"<div class='metric-change {change_class}'>{icon} {change}%</div>"
        
    return f"""
    <div class='metric-card'>
        <div class='metric-title'>{title}</div>
        <div class='metric-value'>{value}</div>
        {change_html}
    </div>
    """

# Function to handle contact information for notifications
def handle_contact_info(email, phone):
    # Store in session state
    st.session_state["patient_email"] = email
    st.session_state["patient_phone"] = phone

    # Prepare medication details if available
    med_details = ""
    if "med_output" in st.session_state:
        output = st.session_state["med_output"]
        med_details = f"""
Your Medication Schedule:
Medication: {output['medication']} {output['dosage']}
Schedule: {output['frequency']} {output['timing']}
Duration: {output['duration']}
Next Refill: {output['refill_date']}
"""

    # ------------------ SMTP Email Part ------------------
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "amennahali8@gmail.com"
    sender_password = "vifntmdgfjgnqzen"

    subject = "Thank you for providing your contact info"
    body = f"""
Hello,

We've received your contact info.
Phone: {phone}
Email: {email}
{med_details}
Best regards,
Your Healthcare App
"""

    msg = MIMEText(body)
    msg["From"] = sender_email
    msg["To"] = email
    msg["Subject"] = subject

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, msg.as_string())
        server.quit()
        st.success("✅ Email sent successfully")
    except Exception as e:
        st.error(f"❌ Email failed: {e}")

    # ------------------ Twilio SMS Part ------------------
    account_sid = "AC2d64558c9c097549ab4bda1adc5cf996"
    auth_token = "1b607c0814fa54d6414f0ceff59bf52d"
    twilio_number = "+12183001925"  # Your Twilio number
    suffix = med_details if med_details else "We'll be in touch."
    sms_body = f"Hi! Your contact info was received. {suffix}"

    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=sms_body,
            from_=twilio_number,
            to=phone
        )
        st.success("✅ SMS sent: " + message.sid)
    except Exception as e:
        st.error(f"❌ SMS failed: {e}")

# Function to generate PDF report
def generate_pdf_report(output):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    c.drawString(50, y, "Structured Clinical Report")
    y -= 30
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Information")
    c.setFont("Helvetica", 12)
    y -= 20
    for key, value in output.get("patient_demographics", {}).items():
        c.drawString(50, y, f"{key.title()}: {value}")
        y -= 20
    if "diagnosis" in output:
        c.drawString(50, y, f"Diagnosis: {output['diagnosis']}")
        y -= 20
    
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Clinical Findings")
    c.setFont("Helvetica", 12)
    y -= 20
    for key, value in output.get("vital_signs", {}).items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20
    if "medications" in output:
        c.drawString(50, y, "Medications:")
        y -= 20
        for med in output["medications"]:
            c.drawString(50, y, f"- {med['name']} {med['dosage']} {med['frequency']}")
            y -= 20
    
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Follow-up Plan")
    c.setFont("Helvetica", 12)
    y -= 20
    follow_up = output.get("follow_up", {})
    c.drawString(50, y, f"When: In {follow_up.get('timeframe', 'N/A')} with {follow_up.get('provider', 'Provider')}")
    y -= 20
    if "action" in output:
        c.drawString(50, y, f"Action: {output['action']}")
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Sidebar for navigation
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🩺 AI Health Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-welcome">Welcome, Dr. Sarah Johnson</div>', unsafe_allow_html=True)
    
    current_time = datetime.now().strftime("%b %d, %Y - %H:%M")
    st.markdown(f"<div style='text-align: center; color: #E0F2FE; margin-bottom: 30px;'>🕒 {current_time}</div>", unsafe_allow_html=True)
    
    st.subheader("Navigation")
    
    selected = st.radio(
        "Select a Feature",
        ["Automated Triage System", "Medication Adherence Assistant", "Mental Health Crisis Detector", "Clinical Report Generator"],
        format_func=lambda x: f"{'🏥' if x == 'Automated Triage System' else '💊' if x == 'Medication Adherence Assistant' else '🧠' if x == 'Mental Health Crisis Detector' else '📝'} {x}",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.subheader("User Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Patients", "127", "+4")
    with col2:
        st.metric("Reports", "43", "+7")
    
    st.markdown("---")
    st.subheader("System Status")
    st.markdown("✅ All systems operational")
    
    st.markdown("---")
    with st.expander("Help & Support"):
        st.markdown("📧 support@aihealthcare.com")
        st.markdown("📞 +1 (800) 555-0123")
        st.markdown("[📚 Documentation](https://example.com)")

# Main content based on selected page
if selected == "Automated Triage System":
    st.markdown('<div class="page-transition">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Automated Triage System")
        st.markdown("Analyze Emergency Room intake notes to prioritize patients based on medical urgency.")
    with col2:
        if lottie_health:
            st_lottie(lottie_health, height=150, key="triage_lottie")
        else:
            st.image("https://via.placeholder.com/150", caption="Triage Animation")
    
    st.markdown(
        f"""
        <div class="metric-container">
            {metric_card("Total Cases", analytics["triage_stats"]["total_cases"], "8.3", True)}
            {metric_card("Critical Cases", analytics["triage_stats"]["urgency_distribution"]["Critical"], "5.2", False)}
            {metric_card("Accuracy Rate", f"{analytics['triage_stats']['accuracy_rate']}%", "1.1", True)}
            {metric_card("Avg. Processing", f"{analytics['triage_stats']['avg_processing_time']}s", "0.3", True)}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patient Clinical Note")
        
        user_input = st.text_area("Enter clinical note (Patient Data Entry):", "Patient reports severe chest pain radiating to left arm, shortness of breath, and dizziness for the past 30 minutes. History of hypertension.", height=150)
        
        if st.button("Analyze Clinical Note"):
            with st.spinner():
                start_time = time.time()
                loading_animation()
                
                prompt = f"""
You are a medical AI assistant specialized in triage analysis. Analyze the following clinical note and provide a structured triage assessment in JSON format. The response must exactly match the structure and detail level of the following example:
```json
{{
  "symptoms": ["chest pain", "radiation to left arm", "shortness of breath", "dizziness"],
  "duration": "30 minutes",
  "medical_history": ["hypertension"],
  "urgency": "Critical",
  "triage_category": "1",
  "recommended_tests": ["ECG", "Cardiac enzymes", "Chest X-ray"],
  "potential_diagnosis": ["Acute Myocardial Infarction", "Angina", "Aortic Dissection"],
  "action": "Cardiac alert sent to EHR and cardiology team notified"
}}
```
Requirements:
- "symptoms": Always return as a list of detailed symptoms. If the clinical note is vague (e.g., "chest pain"), infer additional related symptoms (e.g., "shortness of breath", "dizziness") based on medical likelihood.
- "duration": Provide a reasonable duration if not specified (e.g., "N/A" or an inferred value like "acute").
- "medical_history": Infer a plausible medical history if not specified (e.g., ["hypertension"] for chest pain cases).
- "urgency": Choose from "Critical", "High", "Medium", "Low". Use "Critical" for severe cases like chest pain unless specified otherwise.
- "triage_category": Assign a number from 1 to 5, where 1 is most urgent. Use "1" for critical cases.
- "recommended_tests": Provide a list of relevant tests (e.g., ["ECG", "Cardiac enzymes"] for cardiac issues).
- "potential_diagnosis": Always provide a list of at least two plausible diagnoses (e.g., ["Acute Myocardial Infarction", "Angina"]). Never return an empty list.
- "action": Provide a complete sentence describing the automated action (e.g., "Cardiac alert sent to EHR").
Clinical note: "{user_input}"
Return the response in JSON format, enclosed in triple backticks (```json\n...\n```).
"""
                response_text = call_huggingface_api(prompt, max_length=800)
                if response_text:
                    try:
                        json_str = response_text.strip()
                        if json_str.startswith("```json"):
                            json_str = json_str[7:-3].strip()
                        triage_output = json.loads(json_str)
                        triage_output = normalize_triage_output(triage_output)
                        st.session_state["triage_output"] = triage_output
                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse triage JSON: {str(e)}")
                        st.session_state["triage_output"] = normalize_triage_output({
                            "symptoms": ["chest pain", "radiation to left arm", "shortness of breath", "dizziness"],
                            "duration": "30 minutes",
                            "medical_history": ["hypertension"],
                            "urgency": "Critical",
                            "triage_category": "1",
                            "recommended_tests": ["ECG", "Cardiac enzymes", "Chest X-ray"],
                            "potential_diagnosis": ["Acute Myocardial Infarction", "Angina", "Aortic Dissection"],
                            "action": "Cardiac alert sent to EHR and cardiology team notified"
                        })
                else:
                    st.session_state["triage_output"] = normalize_triage_output({
                        "symptoms": ["chest pain", "radiation to left arm", "shortness of breath", "dizziness"],
                        "duration": "30 minutes",
                        "medical_history": ["hypertension"],
                        "urgency": "Critical",
                        "triage_category": "1",
                        "recommended_tests": ["ECG", "Cardiac enzymes", "Chest X-ray"],
                        "potential_diagnosis": ["Acute Myocardial Infarction", "Angina", "Aortic Dissection"],
                        "action": "Cardiac alert sent to EHR and cardiology team notified"
                    })
                
                st.session_state["triage_processing_time"] = round(time.time() - start_time, 2)
    
    with col2:
        st.subheader("AI Analysis & Action Plan")
        if "triage_output" in st.session_state:
            output = st.session_state["triage_output"]
            
            urgency = output["urgency"]
            if urgency == "Critical":
                urgency_badge = '<span class="status-critical">⚠️ CRITICAL</span>'
            elif urgency == "High":
                urgency_badge = '<span class="status-high">⚠️ HIGH</span>'
            elif urgency == "Medium":
                urgency_badge = '<span class="status-medium">⚠️ MEDIUM</span>'
            else:
                urgency_badge = '<span class="status-low">⚠️ LOW</span>'
                
            st.markdown(f"### Triage Assessment {urgency_badge}", unsafe_allow_html=True)
            
            st.markdown("#### Identified Symptoms")
            symptoms_list = ", ".join([f"**{s}**" for s in output["symptoms"]]) if output["symptoms"] else "None identified"
            st.markdown(symptoms_list)
            
            st.markdown("#### Potential Diagnoses")
            if output["potential_diagnosis"]:
                for diagnosis in output["potential_diagnosis"]:
                    st.markdown(f"- {diagnosis}")
            else:
                st.markdown("No potential diagnoses identified.")
            
            st.markdown("#### Recommended Tests")
            tests_list = ", ".join(output["recommended_tests"]) if output["recommended_tests"] else "None recommended"
            st.markdown(tests_list)
            
            st.markdown("#### Automated Action")
            st.success(f"✅ {output['action']}")
            
            processing_time = st.session_state.get("triage_processing_time", analytics["triage_stats"]["avg_processing_time"])
            st.markdown(f"<div style='text-align: right; color: #64748B; font-size: 12px; margin-top: 20px;'>Processed in {processing_time} seconds</div>", unsafe_allow_html=True)
        else:
            st.info("Enter a clinical note and click 'Analyze Clinical Note' to see AI-generated triage assessment.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("Triage Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            names=list(analytics["triage_stats"]["urgency_distribution"].keys()),
            values=list(analytics["triage_stats"]["urgency_distribution"].values()),
            title="Urgency Distribution",
            color=list(analytics["triage_stats"]["urgency_distribution"].keys()),
            color_discrete_map={
                "Critical": "#EF4444", 
                "High": "#F59E0B", 
                "Medium": "#3B82F6", 
                "Low": "#10B981"
            },
            hole=0.4
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        processing_times = [3.4, 3.3, 3.5, 3.2, 3.1, 2.9, 3.0]
        
        fig = px.line(
            x=days, 
            y=processing_times,
            title="Processing Time Trend (seconds)",
            markers=True
        )
        fig.update_layout(xaxis_title="", yaxis_title="")
        fig.update_traces(line_color="#3B82F6")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "Medication Adherence Assistant":
    st.markdown('<div class="page-transition">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Medication Adherence Assistant")
        st.markdown("Extract medication details from prescriptions to schedule smart reminders and improve patient compliance.")
    with col2:
        if lottie_medication:
            st_lottie(lottie_medication, height=150, key="medication_lottie")
        else:
            st.image("https://via.placeholder.com/150", caption="Medication Animation")
    
    st.markdown(
        f"""
        <div class="metric-container">
            {metric_card("Adherence Improvement", f"{analytics['medication_stats']['adherence_improvement']}%", "2.4", True)}
            {metric_card("Total Reminders", f"{analytics['medication_stats']['total_reminders_sent']:,}", "5.7", True)}
            {metric_card("Response Rate", f"{analytics['medication_stats']['avg_reminder_response']}%", "1.2", True)}
            {metric_card("Active Patients", "843", "3.9", True)}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prescription Information")
        
        user_input = st.text_area("Enter prescription (Patient Data Entry):", "Prescription: Sumatriptan 50mg", height=150)
        
        st.subheader("Patient Contact Information")
        email = st.text_input("Enter patient email:", placeholder="patient@example.com")
        phone = st.text_input("Enter patient phone number:", placeholder="+1234567890")
        
        if st.button("Send notification  and make reminders "):
            if email and phone:
                handle_contact_info(email, phone)
            else:
                st.error("Please provide both email and phone number.")
        
        if st.button("Process Prescription"):
            with st.spinner():
                start_time = time.time()
                loading_animation()
                
                prompt = f"""
You are an AI assistant that generates structured medication adherence plans in JSON format. Analyze the following prescription and provide a structured medication adherence plan exactly matching this structure:
```json
{{
  "medication": "Sumatriptan",
  "dosage": "50mg",
  "frequency": "as needed",
  "timing": "at onset of migraine",
  "duration": "as needed for migraines",
  "patient_concern": "None reported",
  "adherence_risk": "Low",
  "recommendation": "Keep medication accessible + migraine trigger tracking app",
  "refill_date": "N/A - as needed",
  "action": "Patient education on migraine triggers scheduled + medication access reminder set"
}}
```
Requirements:
- Always return a complete JSON structure with all fields, even if the prescription lacks details.
- "medication": Extract the medication name as a string (e.g., "Sumatriptan").
- "dosage": Extract the dosage as a string (e.g., "50mg").
- "frequency": If not specified, infer based on the medication. For Sumatriptan, use "as needed".
- "timing": If not specified, infer based on the medication. For Sumatriptan, use "at onset of migraine".
- "duration": If not specified, infer based on the medication. For Sumatriptan, use "as needed for migraines".
- "patient_concern": If none reported, use "None reported". If a concern is mentioned, extract it.
- "adherence_risk": Assess as "Low", "Medium", or "High". Use "Low" for as-needed medications like Sumatriptan unless concerns are reported.
- "recommendation": Provide a string with a practical suggestion. For Sumatriptan, suggest "Keep medication accessible + migraine trigger tracking app" unless a specific concern suggests otherwise.
- "refill_date": Calculate as YYYY-MM-DD, assuming today is {datetime.now().strftime('%Y-%m-%d')}. For as-needed medications like Sumatriptan, use "N/A - as needed".
- "action": Provide a complete sentence. For Sumatriptan, use "Patient education on migraine triggers scheduled + medication access reminder set" unless a specific concern suggests a different action.
Prescription: "{user_input}"
Return the response in JSON format, enclosed in triple backticks (```json\n...\n```).
"""
                response_text = call_huggingface_api(prompt, max_length=800)
                if response_text:
                    try:
                        json_str = response_text.strip()
                        if json_str.startswith("```json"):
                            json_str = json_str[7:-3].strip()
                        med_output = json.loads(json_str)
                        med_output = normalize_medication_output(med_output)
                        st.session_state["med_output"] = med_output
                    except (json.JSONDecodeError, KeyError) as e:
                        st.error(f"Failed to parse medication JSON: {str(e)}. Using fallback output.")
                        st.session_state["med_output"] = normalize_medication_output({
                            "medication": "Sumatriptan",
                            "dosage": "50mg",
                            "frequency": "as needed",
                            "timing": "at onset of migraine",
                            "duration": "as needed for migraines",
                            "patient_concern": "None reported",
                            "adherence_risk": "Low",
                            "recommendation": "Keep medication accessible + migraine trigger tracking app",
                            "refill_date": "N/A - as needed",
                            "action": "Patient education on migraine triggers scheduled + medication access reminder set"
                        })
                else:
                    st.error("Hugging Face API returned no response. Using fallback output.")
                    st.session_state["med_output"] = normalize_medication_output({
                        "medication": "Sumatriptan",
                        "dosage": "50mg",
                        "frequency": "as needed",
                        "timing": "at onset of migraine",
                        "duration": "as needed for migraines",
                        "patient_concern": "None reported",
                        "adherence_risk": "Low",
                        "recommendation": "Keep medication accessible + migraine trigger tracking app",
                        "refill_date": "N/A - as needed",
                        "action": "Patient education on migraine triggers scheduled + medication access reminder set"
                    })
                
                st.session_state["med_processing_time"] = round(time.time() - start_time, 2)
    
    with col2:
        st.subheader("Medication Schedule & Reminders")
        if "med_output" in st.session_state:
            output = st.session_state["med_output"]
            
            risk = output["adherence_risk"]
            if risk == "High":
                risk_badge = '<span class="status-high">⚠️ HIGH RISK</span>'
            elif risk == "Medium":
                risk_badge = '<span class="status-medium">⚠️ MEDIUM RISK</span>'
            else:
                risk_badge = '<span class="status-low">⚠️ LOW RISK</span>'
                
            st.markdown(f"### Adherence Profile {risk_badge}", unsafe_allow_html=True)
            
            st.markdown("#### Medication Details")
            st.markdown(f"**Medication:** {output['medication']} {output['dosage']}")
            st.markdown(f"**Schedule:** {output['frequency']} {output['timing']}")
            st.markdown(f"**Duration:** {output['duration']}")
            
            st.markdown(f"**Next Refill:** {output['refill_date']}")
            
            st.markdown("#### Adherence Challenge")
            st.markdown(f"_{output['patient_concern']}_")
            
            st.markdown("#### AI Recommendation")
            st.markdown(f"**Solution:** {output['recommendation']}")
            
            st.markdown("#### Automated Action")
            st.success(f"✅ {output['action']}")
            
            if "patient_email" in st.session_state and "patient_phone" in st.session_state:
                st.markdown("#### Patient Contact Details")
                st.markdown(f"**Email:** {st.session_state['patient_email']}")
                st.markdown(f"**Phone:** {st.session_state['patient_phone']}")
                st.info("Use these details to implement notification or medicine time functions.")
            
            processing_time = st.session_state.get("med_processing_time", 2.8)
            st.markdown(f"<div style='text-align: right; color: #64748B; font-size: 12px; margin-top: 20px;'>Processed in {processing_time} seconds</div>", unsafe_allow_html=True)
        else:
            st.info("Enter a prescription and click 'Process Prescription' to see AI-generated medication schedule.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("Medication Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        medications = list(analytics["medication_stats"]["most_common_medications"].keys())
        counts = list(analytics["medication_stats"]["most_common_medications"].values())
        
        fig = px.bar(
            x=medications, 
            y=counts,
            title="Most Common Medications",
            color_discrete_sequence=["#3B82F6"]
        )
        fig.update_layout(xaxis_title="", yaxis_title="Patient Count")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        adherence_rates = [58, 64, 72, 78, 82, 85]
        
        fig = px.line(
            x=months, 
            y=adherence_rates,
            title="Monthly Adherence Rate (%)",
            markers=True
        )
        fig.update_layout(xaxis_title="", yaxis_title="")
        fig.update_traces(line_color="#10B981")
        fig.add_hline(y=60, line_dash="dash", line_color="gray", annotation_text="Target")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "Mental Health Crisis Detector":
    st.markdown('<div class="page-transition">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Mental Health Crisis Detector")
        st.markdown("Monitor patient journals for concerning language and detect potential mental health crises early.")
    with col2:
        if lottie_mental:
            st_lottie(lottie_mental, height=150, key="mental_lottie")
        else:
            st.image("https://via.placeholder.com/150", caption="Mental Health Animation")
    
    st.markdown(
        f"""
        <div class="metric-container">
            {metric_card("High Risk Cases", analytics["mental_health_stats"]["risk_levels"]["High"], "1.5", False)}
            {metric_card("Medium Risk Cases", analytics["mental_health_stats"]["risk_levels"]["Medium"], "2.1", False)}
            {metric_card("Success Rate", f"{analytics['mental_health_stats']['intervention_success_rate']}%", "3.2", True)}
            {metric_card("Avg. Response", f"{analytics['mental_health_stats']['response_time']} hrs", "0.3", True)}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patient Journal Entry")
        
        user_input = st.text_area("Enter journal entry (Patient Data Entry):", "Patient journal: I've been feeling hopeless and overwhelmed lately. I can't sleep, have no appetite, and don't see any point in continuing. Nothing brings me joy anymore.", height=150)
        
        if st.button("Analyze Journal Entry"):
            with st.spinner():
                start_time = time.time()
                loading_animation()
                
                prompt = f"""
You are a mental health AI assistant that assesses risk from patient journals. Analyze the following patient journal entry and provide a structured mental health risk assessment in JSON format exactly matching this structure:
```json
{{
  "risk_phrases": ["hopeless", "overwhelmed", "don't see any point in continuing", "nothing brings me joy"],
  "symptoms": ["insomnia", "appetite loss", "anhedonia", "hopelessness"],
  "risk_level": "High",
  "suicide_risk": "Elevated",
  "recommended_response": "Immediate follow-up within 24 hours",
  "suggested_resources": ["Crisis helpline", "Emergency psychiatric evaluation", "Safety plan development"],
  "action": "Crisis counselor notified and safety check scheduled for today"
}}
```
Requirements:
- "risk_phrases": List phrases indicating risk (e.g., "hopeless"). Return an empty list if none found.
- "symptoms": List symptoms (e.g., "insomnia"). Return an empty list if none identified.
- "risk_level": Assess as "Low", "Medium", or "High". Use "High" for severe cases.
- "suicide_risk": Assess as a string (e.g., "Elevated", "Low to Medium", "Not detected").
- "recommended_response": Provide a timeframe as a string (e.g., "Immediate follow-up within 24 hours"). Use "Routine follow-up" if risk is low.
- "suggested_resources": List at least 2 resources (e.g., "Crisis helpline"). Use general resources if risk is low.
- "action": Provide a complete sentence (e.g., "Crisis counselor notified").
Journal entry: "{user_input}"
Return the response in JSON format, enclosed in triple backticks (```json\n...\n```).
"""
                response_text = call_huggingface_api(prompt, max_length=800)
                if response_text:
                    try:
                        json_str = response_text.strip()
                        if json_str.startswith("```json"):
                            json_str = json_str[7:-3].strip()
                        mental_output = json.loads(json_str)
                        mental_output = normalize_mental_health_output(mental_output)
                        st.session_state["mental_output"] = mental_output
                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse mental health JSON: {str(e)}")
                        st.session_state["mental_output"] = normalize_mental_health_output({
                            "risk_phrases": ["hopeless", "overwhelmed", "don't see any point in continuing", "nothing brings me joy"],
                            "symptoms": ["insomnia", "appetite loss", "anhedonia", "hopelessness"],
                            "risk_level": "High",
                            "suicide_risk": "Elevated",
                            "recommended_response": "Immediate follow-up within 24 hours",
                            "suggested_resources": ["Crisis helpline", "Emergency psychiatric evaluation", "Safety plan development"],
                            "action": "Crisis counselor notified and safety check scheduled for today"
                        })
                else:
                    st.session_state["mental_output"] = normalize_mental_health_output({
                        "risk_phrases": ["hopeless", "overwhelmed", "don't see any point in continuing", "nothing brings me joy"],
                        "symptoms": ["insomnia", "appetite loss", "anhedonia", "hopelessness"],
                        "risk_level": "High",
                        "suicide_risk": "Elevated",
                        "recommended_response": "Immediate follow-up within 24 hours",
                        "suggested_resources": ["Crisis helpline", "Emergency psychiatric evaluation", "Safety plan development"],
                        "action": "Crisis counselor notified and safety check scheduled for today"
                    })
                
                st.session_state["mental_processing_time"] = round(time.time() - start_time, 2)
    
    with col2:
        st.subheader("Risk Assessment & Action Plan")
        if "mental_output" in st.session_state:
            output = st.session_state["mental_output"]
            
            risk = output["risk_level"]
            if risk == "High":
                risk_badge = '<span class="status-high">⚠️ HIGH RISK</span>'
            elif risk == "Medium":
                risk_badge = '<span class="status-medium">⚠️ MEDIUM RISK</span>'
            else:
                risk_badge = '<span class="status-low">✓ LOW RISK</span>'
                
            st.markdown(f"### Risk Assessment {risk_badge}", unsafe_allow_html=True)
            
            if output["risk_phrases"]:
                st.markdown("#### Concerning Language")
                for phrase in output["risk_phrases"]:
                    st.markdown(f"- _{phrase}_")
            
            st.markdown("#### Identified Symptoms")
            symptoms_list = ", ".join([f"**{s}**" for s in output["symptoms"]]) if output["symptoms"] else "None identified"
            st.markdown(symptoms_list)
            
            st.markdown(f"**Suicide Risk:** {output['suicide_risk']}")
            
            st.markdown("#### Recommended Response")
            st.markdown(f"**Timeframe:** {output['recommended_response']}")
            
            st.markdown("#### Suggested Resources")
            for resource in output["suggested_resources"]:
                st.markdown(f"- {resource}")
            
            st.markdown("#### Automated Action")
            st.success(f"✅ {output['action']}")
            
            processing_time = st.session_state.get("mental_processing_time", 3.1)
            st.markdown(f"<div style='text-align: right; color: #64748B; font-size: 12px; margin-top: 20px;'>Processed in {processing_time} seconds</div>", unsafe_allow_html=True)
        else:
            st.info("Enter a journal entry and click 'Analyze Journal Entry' to see AI-generated risk assessment.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("Mental Health Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            names=list(analytics["mental_health_stats"]["risk_levels"].keys()),
            values=list(analytics["mental_health_stats"]["risk_levels"].values()),
            title="Risk Level Distribution",
            color=list(analytics["mental_health_stats"]["risk_levels"].keys()),
            color_discrete_map={
                "High": "#EF4444", 
                "Medium": "#F59E0B", 
                "Low": "#10B981"
            },
            hole=0.4
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        intervention_times = [8.2, 7.5, 6.8, 6.1, 5.7, 5.4]
        
        fig = px.line(
            x=months, 
            y=intervention_times,
            title="Avg. Crisis Response Time (hours)",
            markers=True
        )
        fig.update_layout(xaxis_title="", yaxis_title="")
        fig.update_traces(line_color="#F59E0B")
        fig.add_hline(y=6, line_dash="dash", line_color="gray", annotation_text="Target")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "Clinical Report Generator":
    st.markdown('<div class="page-transition">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Clinical Report Generator")
        st.markdown("Convert unstructured clinical notes into standardized, structured medical reports.")
    with col2:
        if lottie_report:
            st_lottie(lottie_report, height=150, key="report_lottie")
        else:
            st.image("https://via.placeholder.com/150", caption="Report Animation")
    
    st.markdown(
        f"""
        <div class="metric-container">
            {metric_card("Total Reports", analytics["report_stats"]["total_reports"], "12.4", True)}
            {metric_card("Processing Time", f"{analytics['report_stats']['avg_completion_time']}s", "0.5", True)}
            {metric_card("Error Rate", f"{analytics['report_stats']['error_rate']}%", "0.3", True)}
            {metric_card("Satisfaction", f"{analytics['report_stats']['physician_satisfaction']}/5", "0.2", True)}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clinical  Note")
        
        user_input = st.text_area("Enter clinical note (Patient Data Entry):", "Discharge note: Patient is a 65-year-old male admitted for community-acquired pneumonia. Treated with IV antibiotics, now stable. BP 120/80, O2 sat 97% on room air. Continue Amoxicillin 500mg TID for 7 days. Follow up with PCP in 1 week.", height=150)
        
        if st.button("Generate Structured Report"):
            with st.spinner():
                start_time = time.time()
                loading_animation()
                
                prompt = f"""
You are a medical AI assistant that generates structured clinical reports. Convert the following clinical note into a structured medical report in JSON format:
- Extract patient demographics (age, gender, etc.)
- Identify diagnosis or visit type (string)
- Extract treatment (string, if applicable)
- Record vital signs (dictionary)
- List medications (list of dictionaries with name, dosage, frequency, duration)
- Specify follow-up plan (dictionary with provider, timeframe, reason)
- Propose an action (string)
Clinical note: "{user_input}"
Return the response in JSON format, enclosed in triple backticks (```json\n...\n```).
"""
                response_text = call_huggingface_api(prompt, max_length=800)
                if response_text:
                    try:
                        json_str = response_text.strip()
                        if json_str.startswith("```json"):
                            json_str = json_str[7:-3].strip()
                        report_output = json.loads(json_str)
                        st.session_state["report_output"] = report_output
                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse report JSON: {str(e)}")
                        st.session_state["report_output"] = {
                            "patient_demographics": {"age": "65", "gender": "male"},
                            "diagnosis": "Community-acquired pneumonia",
                            "treatment": "IV antibiotics, now transitioned to oral",
                            "vital_signs": {"BP": "120/80", "O2 saturation": "97% on room air"},
                            "medications": [{"name": "Amoxicillin", "dosage": "500mg", "frequency": "TID", "duration": "7 days"}],
                            "follow_up": {"provider": "PCP", "timeframe": "1 week"},
                            "action": "Structured report generated and sent to PCP"
                        }
                else:
                    st.session_state["report_output"] = {
                        "patient_demographics": {"age": "65", "gender": "male"},
                        "diagnosis": "Community-acquired pneumonia",
                        "treatment": "IV antibiotics, now transitioned to oral",
                        "vital_signs": {"BP": "120/80", "O2 saturation": "97% on room air"},
                        "medications": [{"name": "Amoxicillin", "dosage": "500mg", "frequency": "TID", "duration": "7 days"}],
                        "follow_up": {"provider": "PCP", "timeframe": "1 week"},
                        "action": "Structured report generated and sent to PCP"
                    }
                
                st.session_state["report_processing_time"] = round(time.time() - start_time, 2)
    
    with col2:
        st.subheader("Structured Report")
        if "report_output" in st.session_state:
            output = st.session_state["report_output"]
            
            tab1, tab2, tab3 = st.tabs(["Patient Info", "Clinical Data", "Follow-up"])
            
            with tab1:
                st.markdown("#### Patient Information")
                demographics = output["patient_demographics"]
                for key, value in demographics.items():
                    st.markdown(f"**{key.title()}:** {value}")
                
                if "diagnosis" in output:
                    st.markdown(f"**Diagnosis:** {output['diagnosis']}")
                
                if "procedure" in output:
                    st.markdown(f"**Procedure:** {output['procedure']}")
                    st.markdown(f"**Outcome:** {output['procedure_outcome']}")
                
                if "visit_type" in output:
                    st.markdown(f"**Visit Type:** {output['visit_type']}")
            
            with tab2:
                st.markdown("#### Clinical Findings")
                
                st.markdown("**Vital Signs**")
                for key, value in output["vital_signs"].items():
                    st.markdown(f"- {key}: {value}")
                
                if "treatment" in output:
                    st.markdown(f"**Treatment:** {output['treatment']}")
                
                if "medications" in output:
                    st.markdown("**Medications**")
                    for med in output["medications"]:
                        st.markdown(f"- {med['name']} {med['dosage']} {med['frequency']} for {med['duration']}")
                
                if "lab_results" in output:
                    st.markdown(f"**Lab Results:** {output['lab_results']}")
                
                if "diet" in output:
                    st.markdown(f"**Diet:** {output['diet']}")
                
                if "mobility" in output:
                    st.markdown(f"**Mobility:** {output['mobility']}")
            
            with tab3:
                st.markdown("#### Follow-up Plan")
                
                follow_up = output["follow_up"]
                timeframe = follow_up.get("timeframe", "N/A")
                provider = follow_up.get("provider", "Provider")
                reason = follow_up.get("reason", "Routine follow-up")
                
                st.markdown(f"**When:** In {timeframe} with {provider}")
                st.markdown(f"**Reason:** {reason}")
                
                if "discharge_plan" in output:
                    st.markdown(f"**Discharge Plan:** {output['discharge_plan']}")
                
                if "recommendations" in output:
                    st.markdown("**Recommendations:**")
                    for rec in output["recommendations"]:
                        st.markdown(f"- {rec}")
            
            st.markdown("#### Automated Action")
            st.success(f"✅ {output['action']}")
            
            st.markdown("#### Export Options")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Export as PDF"):
                    pdf_buffer = generate_pdf_report(output)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_buffer,
                        file_name="clinical_report.pdf",
                        mime="application/pdf"
                    )
            with col2:
                if st.button("💾 Save to EHR"):
                    st.success("Report saved to EHR (simulated)")
            
            processing_time = st.session_state.get("report_processing_time", analytics["report_stats"]["avg_completion_time"])
            st.markdown(f"<div style='text-align: right; color: #64748B; font-size: 12px; margin-top: 20px;'>Processed in {processing_time} seconds</div>", unsafe_allow_html=True)
        else:
            st.info("Enter a clinical note and click 'Generate Structured Report' to see AI-generated report.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("Report Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days = ["1 week ago", "6 days ago", "5 days ago", "4 days ago", "3 days ago", "2 days ago", "Yesterday"]
        report_counts = [23, 31, 27, 42, 35, 29, 38]
        
        fig = px.bar(
            x=days, 
            y=report_counts,
            title="Daily Report Generation",
            color_discrete_sequence=["#1E40AF"]
        )
        fig.update_layout(xaxis_title="", yaxis_title="Report Count")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        satisfaction_scores = [4.2, 4.3, 4.5, 4.6, 4.65, 4.7]
        
        fig = px.line(
            x=months, 
            y=satisfaction_scores,
            title="Physician Satisfaction Rating (out of 5)",
            markers=True
        )
        fig.update_layout(xaxis_title="", yaxis_title="")
        fig.update_traces(line_color="#10B981")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
