import pandas as pd
from langchain.tools import tool
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr

@tool("code_executor")
def code_executor(code: str, df: pd.DataFrame = None) -> str:
    """Execute Python code safely with dataframe access."""
    try:
        local_vars = {
            "pd": pd,
            "df": df,
            "plt": __import__("matplotlib.pyplot"),
            "sns": __import__("seaborn"),
        }
        
        # Allow built-in functions like print
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            exec(code, None, local_vars)   # None allows normal builtins
        
        result = output.getvalue()
        return result.strip() if result.strip() else "Code executed successfully."
        
    except Exception as e:
        return f"Execution Error:\n{str(e)}\n{traceback.format_exc()}"