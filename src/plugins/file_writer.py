"""
FileWriter plugin for Semantic Kernel.
Provides functionality to write files to the local file system.
"""
import os
from pathlib import Path
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FileWriter:
    """Plugin for writing files to the local file system."""
    
    def __init__(self, output_directory: str = "modified_files"):
        """
        Initialize the FileWriter plugin.
        
        Args:
            output_directory: Directory where files will be written (relative to project root)
        """
        # Get the project root (2 levels up from this file)
        project_root = Path(__file__).parents[2]
        self._directory = os.path.join(project_root, output_directory)
        logger.info(f"FileWriter initialized with directory: {self._directory}")

    @kernel_function(description="Create a file with the specified name and content in the directory.")
    def create_file(
        self,
        file_name: Annotated[str, "The name of the file to create"],
        content: Annotated[str, "The content to write into the file", ""]
    ):
        """
        Create a file with the specified name and content in the directory.
        
        Args:
            file_name (str): The name of the file to create.
            content (str): The content to write into the file. Defaults to an empty string.
            
        Returns:
            str: The path to the created file.
        """
        directory = Path(self._directory)
        file_path = directory / file_name

        # Create all parent directories for the file path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Created file: {file_path}")
        return str(file_path)
