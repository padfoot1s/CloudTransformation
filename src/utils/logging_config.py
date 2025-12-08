"""
Centralized logging configuration for the AST project.
"""
import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Configure logging with a consistent format across the application.
    
    Args:
        level: The logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_logger(name):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: The name for the logger (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
