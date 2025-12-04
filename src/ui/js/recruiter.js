const results = [
  { candidate: "Bhaskar", baseline_score: 72, advanced_score: 85, status: "MATCHED", missing_skills: ["Docker", "Azure"] },
  { candidate: "Santhosh", baseline_score: 50, advanced_score: 62, status: "REJECTED", missing_skills: ["AWS", "ML"] }
];

// Update welcome text safely (no null issue)
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("welcomeText").textContent = "Welcome, Recruiter";
  
  const tableBody = document.getElementById("resultsTable");
  results.forEach(r => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${r.candidate}</td>
      <td>${r.baseline_score}</td>
      <td>${r.advanced_score}</td>
      <td>${r.status}</td>
      <td>${r.missing_skills.join(", ")}</td>
    `;
    tableBody.appendChild(row);
  });
});

// Logout function
function logout() {
  localStorage.clear();
  window.location.href = "login.html";
}
