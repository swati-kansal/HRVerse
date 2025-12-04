// Dummy jobs dataset (10 jobs)
const jobs = [
  { id: 1, title: "Data Analyst", location: "Hyderabad", desc: "SQL, Python, Tableau, Data Visualization" },
  { id: 2, title: "Cloud Engineer", location: "Bangalore", desc: "AWS, Kubernetes, DevOps, Terraform" },
  { id: 3, title: "Marketing Manager", location: "Mumbai", desc: "SEO, Campaigns, Digital Marketing, Branding" },
  { id: 4, title: "HR Specialist", location: "Delhi", desc: "Recruitment, Employee Relations, Payroll, Compliance" },
  { id: 5, title: "Software Developer", location: "Chennai", desc: "Java, Spring Boot, Microservices, REST APIs" },
  { id: 6, title: "Frontend Developer", location: "Pune", desc: "React, Angular, HTML, CSS, JavaScript" },
  { id: 7, title: "Backend Developer", location: "Hyderabad", desc: "Node.js, Express, MongoDB, SQL" },
  { id: 8, title: "Financial Analyst", location: "Gurgaon", desc: "Excel, Power BI, Financial Modeling, Forecasting" },
  { id: 9, title: "UI/UX Designer", location: "Remote", desc: "Figma, Adobe XD, Wireframing, Prototyping" },
  { id: 10, title: "Product Manager", location: "Bangalore", desc: "Agile, Scrum, Product Roadmaps, Stakeholder Management" }
];

// Applications store
let applications = JSON.parse(localStorage.getItem("applications")) || [];

// On load
document.addEventListener("DOMContentLoaded", () => {
  const loggedInUser = localStorage.getItem("loggedInUser") || "candidate";
  document.getElementById("welcomeText").textContent = `Welcome, ${loggedInUser}`;
  updateDashboard();
});

// Update Dashboard
function updateDashboard() {
  if (applications.length > 0) {
    const latest = applications[applications.length - 1];
    document.getElementById("jobTitle").textContent = `Applied Job: ${latest.title}`;
    document.getElementById("jobStatus").textContent = "Applied";
    document.getElementById("nextStep").textContent = "In Review";
  } else {
    document.getElementById("jobTitle").textContent = "No job applied yet.";
    document.getElementById("jobStatus").textContent = "-";
    document.getElementById("nextStep").textContent = "-";
  }
}

// Show Jobs
function showJobs() {
  hideAll();
  document.getElementById("jobsSection").style.display = "block";
  const jobList = document.getElementById("jobList");
  jobList.innerHTML = "";

  jobs.forEach(job => {
    const card = document.createElement("div");
    card.className = "col-md-4 mb-3";
    card.innerHTML = `
      <div class="card shadow-sm p-3 h-100">
        <h5>${job.title}</h5>
        <p><strong>Location:</strong> ${job.location}</p>
        <p><strong>Skills:</strong> ${job.desc}</p>
        <button class="btn btn-accent w-100" onclick="applyJob(${job.id})">Apply</button>
      </div>
    `;
    jobList.appendChild(card);
  });
}

// Apply Job
function applyJob(id) {
  const job = jobs.find(j => j.id === id);
  applications.push(job);
  localStorage.setItem("applications", JSON.stringify(applications));
  alert(`You applied for ${job.title}`);
  updateDashboard();
}

// Show Applications
function showApplications() {
  hideAll();
  document.getElementById("applicationsSection").style.display = "block";

  const list = document.getElementById("applicationsList");
  list.innerHTML = "";
  if (applications.length === 0) {
    list.innerHTML = "<li class='list-group-item'>No applications yet.</li>";
  } else {
    applications.forEach(app => {
      const item = document.createElement("li");
      item.className = "list-group-item";
      item.innerHTML = `<strong>${app.title}</strong> - Status: Applied`;
      list.appendChild(item);
    });
  }
}

// Hide All Sections
function hideAll() {
  document.getElementById("dashboardSection").style.display = "none";
  document.getElementById("jobsSection").style.display = "none";
  document.getElementById("applicationsSection").style.display = "none";
}

// Logout
function logout() {
  localStorage.clear();
  window.location.href = "login.html";
}
