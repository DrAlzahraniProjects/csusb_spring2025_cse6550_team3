## Prerequisites
Install the following before you begin:  
   [Git](https://git-scm.com/)  
   [Docker](https://www.docker.com/)  
   [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)  
  
You must have a [Groq API Key](https://console.groq.com/keys) to run the app.  
If you do not have one, it can be obtained from the Team 3 Discussion on Canvas.  

## To create a simple, Dockerized web application:
1) Clone this GitHub repository
```
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team3.git
```
2) Navigate to the repository
```
cd csusb_spring2025_cse6550_team3 
```
3) Ensure repository is updated to the latest version
```
git pull origin main
```
4) Enable execute permissions for the Docker setup script:
```
chmod +x docker-setup.sh
```
5) Run the script to build and run the Docker image (enter your Groq API Key when prompted):
```
./docker-setup.sh
```
6) Follow the following links to access the app:  
App - http://127.0.0.1:2503/  
Jupyter - http://127.0.0.1:2513/

Alternatively, it can be accessed via the CSE department web server:
https://sec.cse.csusb.edu/team13

### To stop the container from running and remove the container and image:
7) Enable execute permissions for the Docker cleanup script::
```
chmod +x docker-cleanup.sh
```
8) Run the script to stop and remove the Docker image:
```
./docker-cleanup.sh
```
