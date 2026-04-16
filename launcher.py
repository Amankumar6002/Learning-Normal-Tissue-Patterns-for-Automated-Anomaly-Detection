import webbrowser
import os

# Path to the local Medical AI Website
website_path = os.path.join(os.path.dirname(__file__), 'index.html')
website_url = f"file://{website_path}"

def launch_website():
    print("--------------------------------------------------")
    print("Medical AI: VAE Anomaly Detection Project")
    print("--------------------------------------------------")
    print(f"Opening: {website_url}")
    print("Please check your default web browser.")
    print("--------------------------------------------------")
    
    # This command opens the local file in your default browser
    webbrowser.open(website_url)

if __name__ == "__main__":
    launch_website()