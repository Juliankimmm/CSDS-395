import os
import sys
import requests
import json
import getpass
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8585"


class ApiClientCLI:
    def __init__(self):
        self.token = None
        self.headers = {}

    def log(self, message, status="INFO"):
        """Prints a color-coded log message to the console."""
        colors = {
            "INFO": "\033[94m",
            "SUCCESS": "\033[92m",
            "ERROR": "\033[91m",
            "WARNING": "\033[93m",
        }
        reset = "\033[0m"
        print(f"{colors.get(status, '')}{status}: {message}{reset}")

    def _handle_response(self, response):
        """Helper to process and print API responses."""
        if 200 <= response.status_code < 300:
            self.log(f"✓ Request successful (Status: {response.status_code})", status="SUCCESS")
            try:
                # Pretty-print JSON response
                print(json.dumps(response.json(), indent=2))
            except json.JSONDecodeError:
                # Handle empty or non-JSON responses
                self.log("Response was not in JSON format.", status="WARNING")
        else:
            self.log(
                f"✗ Request failed (Status: {response.status_code})",
                status="ERROR",
            )
            self.log(f"Response: {response.text}", status="ERROR")

    def register(self):
        """Prompt for user details and attempt registration."""
        self.log("--- User Registration ---")
        username = input("Enter username: ")
        email = input("Enter email: ")
        password = getpass.getpass("Enter password: ")
        response = requests.post(
            f"{BASE_URL}/register",
            json={"username": username, "email": email, "password": password},
        )
        self._handle_response(response)

    def login(self):
        """Prompt for credentials, log in, and store the token."""
        self.log("--- User Login ---")
        email = input("Enter email: ")
        password = getpass.getpass("Enter password: ")
        response = requests.post(
            f"{BASE_URL}/login", json={"email": email, "password": password}
        )
        if response.status_code == 200:
            self.token = response.json().get("access_token")
            if self.token:
                self.headers = {"Authorization": f"Bearer {self.token}"}
                self.log(
                    "✓ Login successful. Access token stored for subsequent requests.",
                    status="SUCCESS",
                )
                print(f"Access Token: {self.token}")
            else:
                 self.log("✗ Login successful, but no access_token found in response.", status="ERROR")
        else:
            self._handle_response(response)

    def create_contest(self):
        """Prompt for contest details and create it."""
        if not self.token:
            self.log("You must be logged in to create a contest.", status="ERROR")
            return

        self.log("--- Create Contest ---")
        name = input("Enter contest name: ")
        description = input("Enter contest description: ")

        while True:
            try:
                submission_days = int(
                    input("Enter number of days for submissions: ")
                )
                voting_days = int(
                    input("Enter number of days for voting (after submissions end): ")
                )
                break
            except ValueError:
                self.log("Please enter a valid number.", status="ERROR")

        now = datetime.utcnow()
        submission_end = now + timedelta(days=submission_days)
        voting_end = submission_end + timedelta(days=voting_days)

        payload = {
            "name": name,
            "description": description,
            "submission_start_date": now.isoformat(),
            "submission_end_date": submission_end.isoformat(),
            "voting_end_date": voting_end.isoformat(),
        }
        
        # Note: Your original script didn't pass a token here, but creating a
        # contest is often an authenticated action. Add `headers=self.headers` if needed.
        response = requests.post(f"{BASE_URL}/contests", json=payload)
        self._handle_response(response)
        
    def get_contests(self):
        """Fetch and display all contests."""
        self.log("--- Get All Contests ---")
        response = requests.get(f"{BASE_URL}/contests")
        self._handle_response(response)

    def get_contest_detail(self):
        """Prompt for a contest ID and fetch its details."""
        self.log("--- Get Contest Detail ---")
        try:
            contest_id = int(input("Enter Contest ID: "))
            response = requests.get(f"{BASE_URL}/contests/{contest_id}")
            self._handle_response(response)
        except ValueError:
            self.log("Invalid ID. Please enter a number.", status="ERROR")

    def get_submissions(self):
        """Prompt for a contest ID and get its submissions."""
        self.log("--- Get Submissions for a Contest ---")
        try:
            contest_id = int(input("Enter Contest ID: "))
            response = requests.get(f"{BASE_URL}/contests/{contest_id}/submissions")
            self._handle_response(response)
        except ValueError:
            self.log("Invalid ID. Please enter a number.", status="ERROR")

    def create_submission(self):
        """Prompt for contest ID and image path to create a submission."""
        if not self.token:
            self.log("You must be logged in to create a submission.", status="ERROR")
            return

        self.log("--- Create Submission ---")
        try:
            contest_id = int(input("Enter Contest ID to submit to: "))
            image_path = input("Enter path to the image file: ")

            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f)}
                response = requests.post(
                    f"{BASE_URL}/contests/{contest_id}/submissions",
                    files=files,
                    headers=self.headers,
                )
                self._handle_response(response)
        except ValueError:
            self.log("Invalid Contest ID.", status="ERROR")
        except FileNotFoundError:
            self.log(f"File not found at path: {image_path}", status="ERROR")
        except Exception as e:
            self.log(f"An unexpected error occurred: {e}", status="ERROR")


    def vote(self):
        """Prompt for a submission ID and cast a vote."""
        if not self.token:
            self.log("You must be logged in to vote.", status="ERROR")
            return

        self.log("--- Vote for a Submission ---")
        try:
            submission_id = int(input("Enter Submission ID to vote for: "))
            response = requests.post(
                f"{BASE_URL}/submissions/{submission_id}/vote", headers=self.headers
            )
            self._handle_response(response)
        except ValueError:
            self.log("Invalid ID. Please enter a number.", status="ERROR")

def print_menu():
    """Prints the CLI menu."""
    print("\n" + "=" * 30)
    print("      API Test CLI")
    print("=" * 30)
    print("1. Register a new user")
    print("2. Login")
    print("3. Create a new contest")
    print("4. Get all contests")
    print("5. Get contest details by ID")
    print("6. Get submissions for a contest")
    print("7. Create a new submission")
    print("8. Vote for a submission")
    print("0. Exit")
    print("-" * 30)


def main():
    cli = ApiClientCLI()
    while True:
        print_menu()
        choice = input("Enter your choice: ")

        try:
            if choice == "1":
                cli.register()
            elif choice == "2":
                cli.login()
            elif choice == "3":
                cli.create_contest()
            elif choice == "4":
                cli.get_contests()
            elif choice == "5":
                cli.get_contest_detail()
            elif choice == "6":
                cli.get_submissions()
            elif choice == "7":
                cli.create_submission()
            elif choice == "8":
                cli.vote()
            elif choice == "0":
                cli.log("Exiting.", status="INFO")
                break
            else:
                cli.log("Invalid choice, please try again.", status="WARNING")
        except requests.exceptions.ConnectionError:
            cli.log(f"Could not connect to the server at {BASE_URL}. Is it running?", status="ERROR")
        except Exception as e:
            cli.log(f"An unexpected error occurred: {e}", status="ERROR")


if __name__ == "__main__":
    main()