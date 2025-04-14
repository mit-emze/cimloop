import pandas as pd

def helper_funcion(x: int,  y: int = 0) -> int:
    return x ** 2 + y

def generate_pd_table(rows, row_names):
    """
    Generates a pandas dataframe that is neatly displayable as a table in jupyyter notebook
    
    Parameters:
        rows (list of dict): List of dictionaries, where each dictionary represents a row.
        row_names (list of str): List of row labels (must match the number of rows).

    Returns:
        pd.DataFrame: table of given info
    """
    # Convert list of dictionaries to a DataFrame
    df = pd.DataFrame(rows, index=row_names)

    return df
    
    
def generate_latex_table(df, title="Table Title", caption="Table caption", label="table:1"):
    """
    Generate a LaTeX-formatted table from a list of dictionaries, using row names.

    Parameters:
        df (pd.DataFrame): table of info in a pandas dataframe
        title (str): Title of the table.
        caption (str): Caption for the table.
        label (str): Label for referencing in LaTeX.

    Returns:
        str: LaTeX code for the table.
    """

    # Generate LaTeX table code
    latex_table = df.to_latex(index=True, caption=caption, label=label, column_format="|l" + "|c" * len(df.columns) + "|")

    # Add title manually, as pandas doesn't support it
    latex_code = f"\\begin{{table}}[h]\n\\centering\n\\textbf{{{title}}}\n\n{latex_table}\n\\end{{table}}"

    return latex_code