import os # Importing the os module to access environment variables
from flask import Flask # Importing Flask framework to create a web application


# Initializing the Flask application

app = Flask(__name__)

# Defining a route for the root URL ("/")
@app.route("/")
def main():
    return 'Hello World!\n--Team 3' # Returning a simple response

# Running the application
if __name__ == "__main__":
     # Setting the port from environment variable or defaulting to 2503
    port = int(os.environ.get("PORT", 2503))

    # Starting the Flask application with debugging enabled
    # Listening on all network interfaces (0.0.0.0) for accessibility
    app.run(debug=True,host='0.0.0.0',port=port)
