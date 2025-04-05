import sys
import time

class SVALogger:
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        self.log = open(log_path, "w", buffering=1)

    def write(self, message):
        timestamp = time.strftime("%d-%m-%Y %H:%M:%S")
        if message.strip():  # Only log non-empty messages
            if ("% Complete" in message or "Elapsed time" in message
                    or "----" in message or "===" in message or "####" in message):
                formatted = message
            else:
                formatted = f"[{timestamp}] {message}"
            self.terminal.write(formatted)
            self.log.write(formatted)
        else:
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()