from google_auth_oauthlib.flow import InstalledAppFlow

# Scopes required for Gemini API
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

def get_token():
    flow = InstalledAppFlow.from_client_secrets_file(
        'client_secret.json', SCOPES)  # client_secret.json tumhara downloaded OAuth credentials file hai
    creds = flow.run_console()  # ye browser open karega authentication ke liye
    print("Access Token:", creds.token)
    return creds.token

if __name__ == "__main__":
    token = get_token()

