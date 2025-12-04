# 🚀 Running the AI Hiring Portal UI

## Quick Start

### Method 1: Python HTTP Server (Recommended)

```bash
# From the project root directory
python3 run_ui_server.py
```

This will:
- Start a local HTTP server on port 8000
- Automatically open your browser to the login page
- Serve all UI files (HTML, CSS, JS)

### Method 2: Direct Python Server

```bash
# Navigate to the ui directory
cd src/ui

# Start Python's built-in HTTP server
python3 -m http.server 8000

# Then open your browser to:
# http://localhost:8000/login.html
```

### Method 3: VS Code Live Server Extension

1. Install the "Live Server" extension in VS Code
2. Right-click on `src/ui/login.html`
3. Select "Open with Live Server"

## 🔐 Demo Credentials

### Recruiter Login
- **Username**: `recruiter`
- **Password**: `recruiter123`
- **Access**: Can post jobs, view applications, manage candidates

### Candidate Login
- **Username**: `candidate`
- **Password**: `candidate123`
- **Access**: Can view jobs, apply to positions, upload resume

### Admin Login
- **Username**: `admin`
- **Password**: `admin123`
- **Access**: Full system access

## 📁 UI Structure

```
src/ui/
├── login.html          # Login page (entry point)
├── recruiter.html      # Recruiter dashboard
├── candidate.html      # Candidate dashboard
├── navbar.html         # Navigation bar component
├── js/
│   ├── login.js       # Login authentication logic
│   ├── recruiter.js   # Recruiter page logic
│   └── candidate.js   # Candidate page logic
└── style/
    └── style.css      # Custom styles
```

## 🌐 Accessing the Application

Once the server is running, navigate to:
- **Login Page**: http://localhost:8000/login.html
- **Recruiter Dashboard**: http://localhost:8000/recruiter.html (after login)
- **Candidate Dashboard**: http://localhost:8000/candidate.html (after login)

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## 💡 Features

- **Role-based authentication** with dummy credentials
- **Responsive design** using Bootstrap 5
- **Modern UI** with custom styling
- **Client-side routing** between pages
- **Local storage** for session management

## 🔧 Troubleshooting

### Port Already in Use
If port 8000 is already in use, you can specify a different port:

```bash
# Edit run_ui_server.py and change the PORT variable
PORT = 8080  # or any available port
```

### Browser Not Opening Automatically
Manually navigate to: http://localhost:8000/login.html

### Files Not Loading
Make sure you're running the server from the project root directory:
```bash
cd /path/to/ai-hiring-portal
python3 run_ui_server.py
```

## 📝 Notes

- This is a frontend-only demo with dummy authentication
- No backend API is required for basic UI testing
- User sessions are stored in browser localStorage
- For production use, integrate with the backend API in `src/api/`
