"""This module contains functions for interacting with the Q&A database."""

import gspread


def get_google_sheet_data(
    gc: gspread.client.Client, spreadsheet_id: str, sheet_name: str, cell_range: str
) -> list[str] | None:
    """Fetch data from a specified range in a Google Sheet.

    Args:
        gc (gspread.client.Client): The gspread client.
        spreadsheet_id (str): The ID of the spreadsheet.
        sheet_name (str): The name of the sheet.
        cell_range (str): The range of cells to fetch.

    Returns:
        list[str] | None: The values from the specified range or None if an error occurs.
    """
    try:
        # Open the spreadsheet
        sh = gc.open_by_key(spreadsheet_id)
        worksheet = sh.worksheet(sheet_name)

        # Get the values from the specified range
        values = worksheet.get(cell_range)
        return values

    except Exception as e:
        # Handle any errors that occur
        print(f"An error occurred: {e}")
        return None


def write_google_sheet_data(
    gc: gspread.client.Client, spreadsheet_id: str, sheet_name: str, cell_range: str, data: list[str]
) -> dict[str] | None:
    """Write data to a specified range in a Google Sheet.

    Args:
        gc (gspread.client.Client): The gspread client.
        spreadsheet_id (str): The ID of the spreadsheet.
        sheet_name (str): The name of the sheet.
        cell_range (str): The range of cells to update.
        data (list[str]): The data to write.

    Returns:
        dict[str]: A dictionary with the status and updated range, or None if an error occurs.
    """
    try:
        # Open the spreadsheet
        sh = gc.open_by_key(spreadsheet_id)
        worksheet = sh.worksheet(sheet_name)

        # Prepare the data in the required format
        values = [[item] for item in data]

        # Update the specified range with the new values
        worksheet.update(cell_range, values)

        return {"status": "success"}

    except Exception as e:
        # Handle any errors that occur
        print(f"An error occurred: {e}")
        return None


def get_questions() -> list[str]:
    """Retrieve questions from the Google Sheet.

    Returns:
        list[str]: A list of questions from the Google Sheet.
    """
    gc = gspread.service_account()
    spreadsheet_id = "1e_1WA8eUGZddp9NK8ngz_IzZkNby5UcC9JhjMivEvnk"
    sheet_name = "Questions and Responses"
    column_range_read = "B2:B"
    data = get_google_sheet_data(gc, spreadsheet_id, sheet_name, column_range_read)
    return [item[0] for item in data] if data else []


def post_chunks(data: list[str]) -> dict[str] | None:
    """Post chunk data to the Google Sheet.

    Args:
        data (list[str]): The chunk data to write.

    Returns:
        dict[str]: A dictionary with the status and updated range, or None if an error occurs.
    """
    gc = gspread.service_account()
    spreadsheet_id = "1e_1WA8eUGZddp9NK8ngz_IzZkNby5UcC9JhjMivEvnk"
    sheet_name = "Questions and Responses"
    column_range_write = "C2:C"  # Replace with your data
    return write_google_sheet_data(gc, spreadsheet_id, sheet_name, column_range_write, data)


def post_answers(data: list[str]) -> dict[str] | None:
    """Post answers to the Google Sheet.

    Args:
        data (list[str]): The answers to write.

    Returns:
        dict[str]: A dictionary with the status and updated range, or None if an error occurs.
    """
    gc = gspread.service_account()
    spreadsheet_id = "1e_1WA8eUGZddp9NK8ngz_IzZkNby5UcC9JhjMivEvnk"
    sheet_name = "Questions and Responses"
    column_range_write = "D2:D"  # Replace with your data
    return write_google_sheet_data(gc, spreadsheet_id, sheet_name, column_range_write, data)
