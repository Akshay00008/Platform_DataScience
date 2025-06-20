# utility/logger_file.py
import logging

class Logs:
    def __init__(self):
        # Set up the logger
        self.logger = logging.getLogger("InvoiceLogger")
        self.logger.setLevel(logging.DEBUG)  # Set the desired level, e.g., DEBUG, INFO
        
        # Create console handler for logging to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Create formatter for better log output
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to the logger
        self.logger.addHandler(console_handler)

    def info(self, message):
        """Log an info message."""
        self.logger.info(message)
    
    def error(self, message):
        """Log an error message."""
        self.logger.error(message)

    def debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)
