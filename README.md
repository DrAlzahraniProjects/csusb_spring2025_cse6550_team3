## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: [Install Git](https://git-scm.com/) from its official website.
2. **Docker**: [Install Docker](https://www.docker.com) from its official website.
3. **Linux/MacOS**: No extra setup needed.
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable

---

### Step 1: Clone the Repository

Clone the GitHub repository to your local machine:

```
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team3.git
```

### Step 2: Navigate to the Repository

Change to the cloned repository directory:

```
cd csusb_spring2025_cse6550_team3 
```

### Step 3: Pull the Latest Version

Update the repository to the latest version:

```
git pull origin main
```

### Step 4: Set Build Script

Run the setup script to build and start the Docker container:

```
chmod +x docker-setup.sh
```

### Step 5: Run Build Script (enter your Groq API Key when prompted):

```
./docker-setup.sh
```

### Step 6: Access the Chatbot

For Streamlit:

- Once the container starts, Open browser at http://127.0.0.1:2503/team3s25
  
Google colab:

- https://colab.research.google.com/drive/1OICG-s8bcIehEFRvmFZXEZantEJLWayz?usp=sharing

Alternatively, it can be accessed via the CSE department web server:

- https://sec.cse.csusb.edu/team3s25/

### Step 7: Enable execute permissions for the Docker cleanup script:

```
chmod +x docker-cleanup.sh
```

### Step 8: Run the script to stop and remove the Docker image and container :

```
./docker-cleanup.sh
```
