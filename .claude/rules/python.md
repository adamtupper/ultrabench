---
paths:
  - "**/*.py"
---
# Python Coding Standards
## Style and Structure
- Always follow [PEP 8 style conventions](https://peps.python.org/pep-0008/), except for line length. Limit lines to 88 characters for code and 79 characters for docstrings and comments.
- Use explicit type hints for all function signatures.
- Prefer list comprehensions and generator expressions over simple loops where readable.
- Prioritize modern Python features (e.g., f-strings, context managers, comprehensions) over legacy syntax.
- Use `pathlib` for all file system operations instead of `os.path`.
- Add comments sparingly. Use meaningful variable and function names. Add comments only for complex logic that isn't self-explanatory.
- Use `ruff` for linting and formatting via `uv run ruff check --fix --exit-zero`.
- Use `ty` for type checking via `uv run ty check --exit-zero`.

## Documentation
- All modules, classes, and functions must have docstrings in Google style.
- Use triple double quotes (`"""`) for docstrings.
- The summary text should fit on one line and start on the same line as the opening triple quotes (""").
- Include `Args`, `Returns`, and `Raises` sections for every function.
- Don't include type annotations in docstrings.
- Prefer the `|` syntax for union types in type hints (e.g., `str | None` instead of `Optional[str]`).

## Testing
- Use `pytest` for all unit and integration tests via `uv run pytest`.
- Every new feature must include a corresponding test file in the `tests/` directory.
- Use `unittest.mock` or `pytest-mock` for external dependencies.

## Error Handling
- Use specific exceptions (e.g., `ValueError`, `KeyError`) instead of a broad `except Exception`.
- Always include helpful error messages in raised exceptions.

## Required After Each Change
- Always run `ruff` and `ty` to ensure code follows project standards.
- Always run `pytest` to verify changes do not break existing functionality.
- Check the documentation in `docs/` and the `README.md` file to ensure it is up to date with any changes made. Update as necessary.

## Examples
A good docstring:

```python
def calculate_area(width: float, height: float) -> float:
    """Calculate the area of a rectangle.
    
    Args:
        width: The width of the rectangle.
        height: The height of the rectangle.

    Returns:
        The total area of the rectangle.

    Raises:
        ValueError: If width or height is negative.
    """
    if width < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative.")
    
    return width * height
```
