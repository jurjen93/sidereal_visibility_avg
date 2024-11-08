from sys import stdout

def print_progress_bar(index, total, bar_length=50):
    """
    Prints a progress bar to the console.

    :param:
        - index: the current index (0-based) in the iteration.
        - total: the total number of indices.
        - bar_length: the character length of the progress bar (default 50).
    """

    percent_complete = (index + 1) / total
    filled_length = int(bar_length * percent_complete)
    bar = "█" * filled_length + '-' * (bar_length - filled_length)
    stdout.write(f'\rProgress: |{bar}| {percent_complete * 100:.1f}% Complete')
    stdout.flush()  # Important to ensure the progress bar is updated in place

    # Print a new line on completion
    if index == total - 1:
        print()