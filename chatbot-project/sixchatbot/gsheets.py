"""Contains QADatabase a class for interacting with the Q&A Google Sheets database."""

import gspread


class QADatabase:
    """Class for interacting with the Q&A Google Sheets database."""

    def __init__(self, spreadsheet_id: str, sheet_name: str) -> None:
        """Initialize the QADatabase with a Google Sheets client and a spreadsheet ID.

        Args:
            spreadsheet_id (str): The ID of the spreadsheet.
            sheet_name (str): The name of the sheet.
        """
        self.gc = gspread.service_account()
        self.spreadsheet_id = spreadsheet_id
        self.sheet_name = sheet_name

    def get_google_sheet_data(self, cell_range: str) -> list[str] | None:
        """Fetch data from a specified range in a Google Sheet.

        Args:
            cell_range (str): The range of cells to fetch.

        Returns:
            list[str] | None: The values from the specified range or None if an error occurs.
        """
        try:
            sh = self.gc.open_by_key(self.spreadsheet_id)
            worksheet = sh.worksheet(self.sheet_name)

            values = worksheet.get(cell_range)
            return values

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def write_google_sheet_data(self, cell_range: str, data: list[str]) -> dict[str] | None:
        """Write data to a specified range in a Google Sheet.

        Args:
            sheet_name (str): The name of the sheet.
            cell_range (str): The range of cells to update.
            data (list[str]): The data to write.

        Returns:
            dict[str]: A dictionary with the status and updated range, or None if an error occurs.
        """
        try:
            sh = self.gc.open_by_key(self.spreadsheet_id)
            worksheet = sh.worksheet(self.sheet_name)

            values = [[item] for item in data]
            worksheet.update(cell_range, values)

            return {"status": "success"}

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_questions(self) -> list[str]:
        """Retrieve questions from the Google Sheet.

        Returns:
            list[str]: A list of questions from the Google Sheet.
        """
        column_range_read = "B2:B21"
        data = self.get_google_sheet_data(column_range_read)
        return [item[0] for item in data] if data else []

    def post_chunks(self, data: list[str]) -> dict[str] | None:
        """Post chunk data to the Google Sheet.

        Args:
            data (list[str]): The chunk data to write.

        Returns:
            dict[str]: A dictionary with the status and updated range, or None if an error occurs.
        """
        column_range_write = "D2:D"
        return self.write_google_sheet_data(column_range_write, data)

    def post_answers(self, data: list[str]) -> dict[str] | None:
        """Post answers to the Google Sheet.

        Args:
            data (list[str]): The answers to write.

        Returns:
            dict[str]: A dictionary with the status and updated range, or None if an error occurs.
        """
        column_range_write = "E2:E"
        return self.write_google_sheet_data(column_range_write, data)
