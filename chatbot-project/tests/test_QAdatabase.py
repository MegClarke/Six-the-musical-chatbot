import unittest
from unittest.mock import MagicMock, patch

from sixchatbot.gsheets import QADatabase


class TestQADatabase(unittest.TestCase):
    @patch("gspread.service_account")
    def setUp(self, mock_service_account):
        """Set up a QADatabase instance with mocked Google Sheets client."""
        self.mock_gc = mock_service_account.return_value
        self.spreadsheet_id = "test_spreadsheet_id"
        self.sheet_name = "test_sheet_name"
        self.qadb = QADatabase(self.spreadsheet_id, self.sheet_name)

    def test_initialization(self):
        """Test that the QADatabase is initialized with the correct spreadsheet ID and sheet name."""
        self.assertEqual(self.qadb.spreadsheet_id, self.spreadsheet_id)
        self.assertEqual(self.qadb.sheet_name, self.sheet_name)
        self.assertEqual(self.qadb.gc, self.mock_gc)

    @patch("sixchatbot.gsheets.gspread.service_account")
    def test_get_google_sheet_data_success(self, mock_service_account):
        """Test fetching data from Google Sheets successfully."""
        mock_worksheet = self.mock_gc.open_by_key.return_value.worksheet.return_value
        mock_worksheet.get.return_value = [["Question 1"], ["Question 2"]]

        result = self.qadb.get_google_sheet_data("B2:B21")
        self.assertEqual(result, [["Question 1"], ["Question 2"]])

        self.mock_gc.open_by_key.assert_called_once_with(self.spreadsheet_id)
        mock_worksheet.get.assert_called_once_with("B2:B21")

    @patch("sixchatbot.gsheets.gspread.service_account")
    def test_get_google_sheet_data_failure(self, mock_service_account):
        """Test handling errors when fetching data from Google Sheets."""
        self.mock_gc.open_by_key.side_effect = Exception("Some error")

        result = self.qadb.get_google_sheet_data("B2:B21")
        self.assertIsNone(result)

    @patch("sixchatbot.gsheets.gspread.service_account")
    def test_write_google_sheet_data_success(self, mock_service_account):
        """Test writing data to Google Sheets successfully."""
        mock_worksheet = self.mock_gc.open_by_key.return_value.worksheet.return_value

        result = self.qadb.write_google_sheet_data("D2:D", ["Chunk 1", "Chunk 2"])
        self.assertEqual(result, {"status": "success"})

        self.mock_gc.open_by_key.assert_called_once_with(self.spreadsheet_id)
        mock_worksheet.update.assert_called_once_with("D2:D", [["Chunk 1"], ["Chunk 2"]])

    @patch("sixchatbot.gsheets.gspread.service_account")
    def test_write_google_sheet_data_failure(self, mock_service_account):
        """Test handling errors when writing data to Google Sheets."""
        self.mock_gc.open_by_key.side_effect = Exception("Some error")

        result = self.qadb.write_google_sheet_data("D2:D", ["Chunk 1", "Chunk 2"])
        self.assertIsNone(result)

    @patch("sixchatbot.gsheets.gspread.service_account")
    def test_get_questions(self, mock_service_account):
        """Test retrieving questions from Google Sheets."""
        self.qadb.get_google_sheet_data = MagicMock(return_value=[["Question 1"], ["Question 2"]])

        result = self.qadb.get_questions()
        self.assertEqual(result, ["Question 1", "Question 2"])

        self.qadb.get_google_sheet_data.assert_called_once_with("B2:B21")

    @patch("sixchatbot.gsheets.gspread.service_account")
    def test_post_chunks(self, mock_service_account):
        """Test posting chunks to Google Sheets."""
        self.qadb.write_google_sheet_data = MagicMock(return_value={"status": "success"})

        result = self.qadb.post_chunks(["Chunk 1", "Chunk 2"])
        self.assertEqual(result, {"status": "success"})

        self.qadb.write_google_sheet_data.assert_called_once_with("D2:D", ["Chunk 1", "Chunk 2"])

    @patch("sixchatbot.gsheets.gspread.service_account")
    def test_post_answers(self, mock_service_account):
        """Test posting answers to Google Sheets."""
        self.qadb.write_google_sheet_data = MagicMock(return_value={"status": "success"})

        result = self.qadb.post_answers(["Answer 1", "Answer 2"])
        self.assertEqual(result, {"status": "success"})

        self.qadb.write_google_sheet_data.assert_called_once_with("E2:E", ["Answer 1", "Answer 2"])
