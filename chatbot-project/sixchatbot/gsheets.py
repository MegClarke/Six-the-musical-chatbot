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
        sh = gc.open_by_key(spreadsheet_id)
        worksheet = sh.worksheet(sheet_name)

        values = worksheet.get(cell_range)
        return values

    except Exception as e:
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
        sh = gc.open_by_key(spreadsheet_id)
        worksheet = sh.worksheet(sheet_name)

        values = [[item] for item in data]
        worksheet.update(values, cell_range)

        return {"status": "success"}

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_questions(sheetname: str) -> list[str]:
    """Retrieve questions from the Google Sheet.

    Args:
        sheetname (str): The name of the sheet to read from

    Returns:
        list[str]: A list of questions from the Google Sheet.
    """
    gc = gspread.service_account()
    spreadsheet_id = "1e_1WA8eUGZddp9NK8ngz_IzZkNby5UcC9JhjMivEvnk"
    sheet_name = sheetname
    column_range_read = "B2:B"
    data = get_google_sheet_data(gc, spreadsheet_id, sheet_name, column_range_read)
    return [item[0] for item in data] if data else []


def post_chunks(sheetname: str, data: list[str]) -> dict[str] | None:
    """Post chunk data to the Google Sheet.

    Args:
        sheetname (str): The name of the sheet to write to.
        data (list[str]): The chunk data to write.

    Returns:
        dict[str]: A dictionary with the status and updated range, or None if an error occurs.
    """
    gc = gspread.service_account()
    spreadsheet_id = "1e_1WA8eUGZddp9NK8ngz_IzZkNby5UcC9JhjMivEvnk"
    sheet_name = sheetname
    column_range_write = "D2:D"
    return write_google_sheet_data(gc, spreadsheet_id, sheet_name, column_range_write, data)


def post_answers(sheetname: str, data: list[str]) -> dict[str] | None:
    """Post answers to the Google Sheet.

    Args:
        sheetname (str): The name of the sheet to write to.
        data (list[str]): The answers to write.

    Returns:
        dict[str]: A dictionary with the status and updated range, or None if an error occurs.
    """
    gc = gspread.service_account()
    spreadsheet_id = "1e_1WA8eUGZddp9NK8ngz_IzZkNby5UcC9JhjMivEvnk"
    sheet_name = sheetname
    column_range_write = "E2:E"
    return write_google_sheet_data(gc, spreadsheet_id, sheet_name, column_range_write, data)
