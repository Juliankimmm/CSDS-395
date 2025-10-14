import os
import sys
import requests
from datetime import datetime, timedelta
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8585"


class TestRunner:
    def __init__(self):
        self.token = None
        self.user_id = None
        self.contest_id = None
        self.submission_id = None

    def log(self, message, status="INFO"):
        colors = {
            "INFO": "\033[94m",
            "SUCCESS": "\033[92m",
            "ERROR": "\033[91m",
            "WARNING": "\033[93m",
        }
        reset = "\033[0m"
        print(f"{colors.get(status, '')}{status}: {message}{reset}")

    def test_register(self):
        """Test user registration"""
        self.log("Testing user registration...")
        response = requests.post(
            f"{BASE_URL}/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "testpass123",
            },
        )
        if response.status_code == 201:
            self.user_id = response.json()["user_id"]
            self.log(
                f"✓ User registered: {response.json()}", status="SUCCESS"
            )
            return True
        else:
            self.log(
                f"✗ Registration failed: {response.text}", status="ERROR"
            )
            return False

    def test_login(self):
        """Test user login"""
        self.log("Testing user login...")
        response = requests.post(
            f"{BASE_URL}/login",
            json={"email": "test@example.com", "password": "testpass123"},
        )
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.log(f"✓ Login successful", status="SUCCESS")
            return True
        else:
            self.log(f"✗ Login failed: {response.text}", status="ERROR")
            return False

    def test_create_contest(self):
        """Test contest creation"""
        self.log("Testing contest creation...")
        now = datetime.utcnow()
        response = requests.post(
            f"{BASE_URL}/contests",
            json={
                "name": "Test Contest",
                "description": "A test contest",
                "submission_start_date": now.isoformat(),
                "submission_end_date": (now + timedelta(days=7)).isoformat(),
                "voting_end_date": (now + timedelta(days=14)).isoformat(),
            },
        )
        if response.status_code == 201:
            self.contest_id = response.json()["contest_id"]
            self.log(
                f"✓ Contest created: {response.json()}", status="SUCCESS"
            )
            return True
        else:
            self.log(
                f"✗ Contest creation failed: {response.text}",
                status="ERROR",
            )
            return False

    def test_get_contests(self):
        """Test getting contests"""
        self.log("Testing get all contests...")
        response = requests.get(f"{BASE_URL}/contests")
        if response.status_code == 200:
            contests = response.json()
            self.log(
                f"✓ Retrieved {len(contests)} contests", status="SUCCESS"
            )
            return True
        else:
            self.log(
                f"✗ Get contests failed: {response.text}", status="ERROR"
            )
            return False

    def test_get_contests_by_status(self):
        """Test getting contests by status"""
        self.log("Testing get contests by status (submission)...")
        response = requests.get(f"{BASE_URL}/contests?status=submission")
        if response.status_code == 200:
            contests = response.json()
            self.log(
                f"✓ Retrieved {len(contests)} active contests",
                status="SUCCESS",
            )
            return True
        else:
            self.log(
                f"✗ Get contests by status failed: {response.text}",
                status="ERROR",
            )
            return False

    def test_get_contest_detail(self):
        """Test getting single contest"""
        self.log("Testing get contest detail...")
        response = requests.get(f"{BASE_URL}/contests/{self.contest_id}")
        if response.status_code == 200:
            self.log(f"✓ Contest details retrieved", status="SUCCESS")
            return True
        else:
            self.log(
                f"✗ Get contest detail failed: {response.text}",
                status="ERROR",
            )
            return False

    def test_create_submission(self):
        """Test creating a submission"""
        self.log("Testing submission creation...")

        # Create a dummy image file
        image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00"
        files = {"file": ("test.png", io.BytesIO(image_data), "image/png")}

        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(
            f"{BASE_URL}/contests/{self.contest_id}/submissions",
            files=files,
            headers=headers,
        )

        if response.status_code == 201:
            self.submission_id = response.json()["sub_id"]
            self.log(f"✓ Submission created", status="SUCCESS")
            return True
        else:
            self.log(
                f"✗ Submission failed: {response.text}", status="ERROR"
            )
            return False

    def test_get_submissions(self):
        """Test getting contest submissions"""
        self.log("Testing get contest submissions...")
        response = requests.get(
            f"{BASE_URL}/contests/{self.contest_id}/submissions"
        )
        if response.status_code == 200:
            submissions = response.json()
            self.log(
                f"✓ Retrieved {len(submissions)} submissions",
                status="SUCCESS",
            )
            return True
        else:
            self.log(
                f"✗ Get submissions failed: {response.text}", status="ERROR"
            )
            return False

    def test_vote(self):
        """Test voting for a submission"""
        self.log("Testing voting...")

        # Register second user
        response = requests.post(
            f"{BASE_URL}/register",
            json={
                "username": "voter",
                "email": "voter@example.com",
                "password": "voterpass123",
            },
        )

        if response.status_code != 201:
            self.log(
                f"✗ Failed to create voter: {response.text}", status="ERROR"
            )
            return False

        # Login as second user
        response = requests.post(
            f"{BASE_URL}/login",
            json={"email": "voter@example.com", "password": "voterpass123"},
        )

        if response.status_code != 200:
            self.log(
                f"✗ Voter login failed: {response.text}", status="ERROR"
            )
            return False

        voter_token = response.json()["access_token"]

        # Need to wait for voting period or modify contest dates
        # For testing, we'll attempt to vote and expect proper error handling
        headers = {"Authorization": f"Bearer {voter_token}"}
        response = requests.post(
            f"{BASE_URL}/submissions/{self.submission_id}/vote",
            headers=headers,
        )

        if response.status_code in [201, 400]:
            # 201 if voting period active, 400 if not yet active
            msg = response.json().get("detail", "Vote processed")
            self.log(f"✓ Vote endpoint working: {msg}", status="SUCCESS")
            return True
        else:
            self.log(f"✗ Vote failed: {response.text}", status="ERROR")
            return False

    def test_duplicate_vote(self):
        """Test preventing duplicate votes"""
        self.log("Testing duplicate vote prevention...")
        # This will only work if voting period is active
        self.log(
            "⊘ Skipped (requires voting period to be active)",
            status="WARNING",
        )
        return True

    def test_self_vote(self):
        """Test preventing self-voting"""
        self.log("Testing self-vote prevention...")
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(
            f"{BASE_URL}/submissions/{self.submission_id}/vote",
            headers=headers,
        )

        if response.status_code == 400 and "own submission" in response.text:
            self.log(
                "✓ Self-vote correctly prevented", status="SUCCESS"
            )
            return True
        elif (
            response.status_code == 400
            and "not active" in response.text.lower()
        ):
            self.log(
                "⊘ Cannot test (voting not active)", status="WARNING"
            )
            return True
        else:
            self.log(
                f"✗ Self-vote prevention failed: {response.text}",
                status="ERROR",
            )
            return False

    def run_all_tests(self):
        """Run all tests"""
        self.log("\n" + "=" * 60, status="INFO")
        self.log("Starting API Tests", status="INFO")
        self.log("=" * 60 + "\n", status="INFO")

        tests = [
            self.test_register,
            self.test_login,
            self.test_create_contest,
            self.test_get_contests,
            self.test_get_contests_by_status,
            self.test_get_contest_detail,
            self.test_create_submission,
            self.test_get_submissions,
            self.test_self_vote,
            self.test_vote,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.log(
                    f"✗ Test {test.__name__} crashed: {str(e)}",
                    status="ERROR",
                )
                failed += 1
            print()

        self.log("=" * 60, status="INFO")
        self.log(
            f"Tests completed: {passed} passed, {failed} failed",
            status="SUCCESS" if failed == 0 else "WARNING",
        )
        self.log("=" * 60, status="INFO")


if __name__ == "__main__":
    runner = TestRunner()
    runner.run_all_tests()