// Dummy JSON credentials with roles
const users = [
  { username: "admin", password: "admin123", role: "admin" },
  { username: "recruiter", password: "recruiter123", role: "recruiter" },
  { username: "candidate", password: "candidate123", role: "candidate" }
];

document.getElementById("loginForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const uname = document.getElementById("username").value.trim();
  const pwd = document.getElementById("password").value.trim();
  const errorMsg = document.getElementById("errorMsg");

  // Check credentials
  const user = users.find(u => u.username === uname && u.password === pwd);

  if (user) {
    localStorage.setItem("loggedInUser", uname);
    localStorage.setItem("userRole", user.role);

    // Redirect based on role
    if (user.role === "recruiter") {
      window.location.href = "recruiter.html";
    } else if (user.role === "candidate") {
      window.location.href = "candidate.html";
    } else if (user.role === "admin") {
      window.location.href = "admin.html";
    }
  } else {
    errorMsg.textContent = "Invalid username or password!";
  }
});
