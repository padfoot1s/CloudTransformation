"""
AgentFactory - Factory pattern for creating and configuring Azure AI Agents.
"""
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from semantic_kernel import Kernel
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread
from azure.ai.projects.models import FilePurpose, FileSearchTool
from datetime import timedelta

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AgentFactory:
    """Factory class for creating and configuring Azure AI Agents."""
    
    @staticmethod
    def detect_language(folder_path: str) -> Optional[str]:
        """
        Detect the primary language in the codebase by scanning for .java or .cs files.
        
        Args:
            folder_path: Path to the folder to scan
            
        Returns:
            'java', 'csharp', or None if no supported language is found
        """
        logger.info(f"Detecting language in folder: {folder_path}")
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.java'):
                    logger.info("Detected language: JAVA")
                    return 'java'
                elif file.endswith('.cs'):
                    logger.info("Detected language: C#")
                    return 'csharp'
        logger.warning("No Java or C# files found in the input folder")
        return None
    
    @staticmethod
    def get_java_prompt() -> str:
        """Get the transformation prompt for Java codebases."""
        return """
Analyze the provided Java AWS Lambda codebase. Your goal is to transform this into a deployment-ready Azure Function App project using Java (Java 17 or 21).

IMPORTANT: Only convert production Lambda handler files. Do NOT convert test files.

**CRITICAL: IN-PLACE CONVERSION**
1.  **File Preservation**: Convert each file in-place, maintaining the EXACT same file path and name.
2.  **Logic Migration**: MIGRATE AWS Lambda logic to Azure Function logic within the same file.
3.  **Structure Preservation**: Keep the original package structure and class names.

**Service Mapping**:
*   **AWS Lambda Handler** -> **Azure Function with @FunctionName annotation**
*   **EC2** -> **Azure Virtual Machines** (Use `com.azure.resourcemanager:azure-resourcemanager-compute`)
*   **S3** -> **Azure Blob Storage** (Use `com.azure:azure-storage-blob`)
*   **DynamoDB** -> **Azure Cosmos DB** (Use `com.azure:azure-cosmos`)
*   **SQS** -> **Azure Queue Storage** (Use `com.azure:azure-storage-queue`)

**Concept Mapping**:
*   **Lambda Context** -> **ExecutionContext**
*   **APIGatewayProxyRequestEvent** -> **HttpRequestMessage<Optional<String>>**
*   **APIGatewayProxyResponseEvent** -> **HttpResponseMessage**
*   **AMI IDs** -> **Azure Image References**
*   **Instance Types** -> **Azure VM Sizes**
*   **Spot Instances** -> **Azure Spot VMs** (`Priority = VirtualMachinePriority.SPOT`)

**SDK Usage**:
*   Remove `aws-java-sdk-*` dependencies.
*   Add `com.azure:azure-identity`, `com.azure.resourcemanager:azure-resourcemanager`, etc.
*   Add `com.microsoft.azure.functions:azure-functions-java-library`

**Conversion Instructions**:
1. Convert the Lambda handler class to an Azure Function class
2. Replace AWS SDK calls with Azure SDK equivalents
3. Update method signatures to use Azure Function annotations (@FunctionName, @HttpTrigger, etc.)
4. Maintain the same file path and class name
5. Keep all business logic in the same file

You must also generate configuration files:
*   `pom.xml`: Complete Maven project file with Azure Functions dependencies
*   `host.json`: Azure Functions host configuration
*   `local.settings.json`: Local development settings

Use the `file_writer` tool to write each file to the local file system.
Do not just output code snippets; write the full file contents.
"""
    
    @staticmethod
    def get_csharp_prompt() -> str:
        """Get the transformation prompt for C# codebases."""
        return """
Analyze the provided C# AWS Lambda codebase. Your goal is to transform this into a deployment-ready Azure Function App project using C# (.NET 6 or .NET 8).

IMPORTANT: Only convert production Lambda handler files. Do NOT convert test files.

**CRITICAL: IN-PLACE CONVERSION**
1.  **File Preservation**: Convert each file in-place, maintaining the EXACT same file path and name.
2.  **Logic Migration**: MIGRATE AWS Lambda logic to Azure Function logic within the same file.
3.  **Structure Preservation**: Keep the original namespace and class names.

**Service Mapping**:
*   **AWS Lambda Function** -> **Azure Function with [FunctionName] attribute**
*   **EC2** -> **Azure Virtual Machines** (Use `Azure.ResourceManager.Compute`)
*   **S3** -> **Azure Blob Storage** (Use `Azure.Storage.Blobs`)
*   **DynamoDB** -> **Azure Cosmos DB** (Use `Microsoft.Azure.Cosmos`)
*   **SQS** -> **Azure Queue Storage** (Use `Azure.Storage.Queues`)

**Concept Mapping**:
*   **ILambdaContext** -> **ExecutionContext**
*   **APIGatewayProxyRequest** -> **HttpRequest**
*   **APIGatewayProxyResponse** -> **IActionResult** or **HttpResponseData**
*   **AMI IDs** -> **Azure Image References**
*   **Instance Types** -> **Azure VM Sizes**
*   **Spot Instances** -> **Azure Spot VMs** (`Priority = VirtualMachinePriority.Spot`)

**SDK Usage**:
*   Remove `AWSSDK.*` packages.
*   Add `Azure.Identity`, `Azure.ResourceManager`, `Azure.ResourceManager.Compute`, etc.
*   Add `Microsoft.Azure.Functions.Worker` or `Microsoft.NET.Sdk.Functions`

**Conversion Instructions**:
1. Convert the Lambda function class to an Azure Function class
2. Replace AWS SDK calls with Azure SDK equivalents
3. Update method signatures to use Azure Function attributes ([FunctionName], [HttpTrigger], etc.)
4. Maintain the same file path, namespace, and class name
5. Keep all business logic in the same file
6. Add dependency injection if needed (constructor injection)

You must also generate configuration files:
*   `*.csproj`: Complete project file with Azure Functions SDK
*   `host.json`: Azure Functions host configuration
*   `local.settings.json`: Local development settings
*   `Program.cs`: DI configuration (if using isolated worker model)

Use the `file_writer` tool to write each file to the local file system.
Do not just output code snippets; write the full file contents.
"""
    
    @staticmethod
    def get_prompt(language: str) -> str:
        """
        Get the appropriate prompt for the detected language.
        
        Args:
            language: 'java' or 'csharp'
            
        Returns:
            The language-specific transformation prompt
        """
        if language == 'java':
            return AgentFactory.get_java_prompt()
        elif language == 'csharp':
            return AgentFactory.get_csharp_prompt()
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    @staticmethod
    def read_extra_context(input_folder: str, language: str) -> str:
        """
        Read configuration files and README.md to provide extra context.
        
        Args:
            input_folder: Path to the input folder
            language: 'java' or 'csharp'
            
        Returns:
            Extra context string with file contents
        """
        extra_context = ""
        
        # For C#, find .csproj files (skip test projects)
        if language == 'csharp':
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.endswith('.csproj'):
                        file_path = os.path.join(root, file)
                        # Skip test project files
                        if '/test/' in file_path.lower() or '/tests/' in file_path.lower():
                            logger.info(f"Skipping test project: {file_path}")
                            continue
                        try:
                            with open(file_path, "r") as f:
                                extra_context += f"\n\nHere is the content of {file}:\n{f.read()}"
                                logger.info(f"Read configuration file: {file}")
                        except FileNotFoundError:
                            pass
        else:  # Java
            try:
                pom_path = os.path.join(input_folder, "pom.xml")
                with open(pom_path, "r") as f:
                    extra_context += f"\n\nHere is the content of pom.xml:\n{f.read()}"
                    logger.info("Read pom.xml configuration file")
            except FileNotFoundError:
                logger.warning("pom.xml not found")
        
        # Read README.md for both languages
        try:
            readme_path = os.path.join(input_folder, "README.md")
            with open(readme_path, "r") as f:
                extra_context += f"\n\nHere is the content of README.md:\n{f.read()}"
                logger.info("Read README.md file")
        except FileNotFoundError:
            logger.warning("README.md not found")
        
        return extra_context
    
    @staticmethod
    async def upload_files(
        client,
        input_folder: str,
        language: str
    ) -> Tuple[List[str], Dict[str, Dict[str, str]], List[str]]:
        """
        Upload source files to Azure and create file mappings with metadata.
        
        Args:
            client: Azure AI client
            input_folder: Path to the input folder
            language: 'java' or 'csharp'
            
        Returns:
            Tuple of (uploaded_file_ids, file_metadata_map, temp_files)
            file_metadata_map contains: {
                'relative_path': {
                    'file_id': str,
                    'filename': str,
                    'relative_path': str,
                    'directory': str,
                    'namespace_or_package': str
                }
            }
        """
        file_extension = '.java' if language == 'java' else '.cs'
        uploaded_files = []
        file_metadata_map = {}  # Map relative_path to metadata dict
        temp_files = []  # Track temporary files for cleanup
        
        logger.info(f"Starting file upload for {language} files")
        
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith(file_extension):
                    # Skip test files
                    if '/test/' in file_path.lower() or '/tests/' in file_path.lower():
                        logger.info(f"Skipping test file: {file_path}")
                        continue
                    
                    # Calculate relative path from input_folder
                    relative_path = os.path.relpath(file_path, input_folder)
                    directory = os.path.dirname(relative_path)
                    
                    # Extract namespace/package from file path
                    namespace_or_package = AgentFactory._extract_namespace_from_path(
                        relative_path, language
                    )
                    
                    upload_path = file_path
                    search_filename = file  # Default to just filename
                    
                    # Workaround: Azure doesn't support .cs extension, so create a .txt copy
                    if language == 'csharp':
                        temp_path = file_path + '.txt'
                        shutil.copy2(file_path, temp_path)
                        upload_path = temp_path
                        temp_files.append(temp_path)
                        search_filename = file + '.txt'  # Agent should search for .txt version
                        logger.debug(f"Created temporary file {temp_path} for upload")
                    
                    _file = await client.agents.upload_file_and_poll(
                        file_path=upload_path,
                        purpose=FilePurpose.AGENTS
                    )
                    uploaded_files.append(_file.id)
                    
                    # Store comprehensive metadata
                    file_metadata_map[relative_path] = {
                        'file_id': _file.id,
                        'filename': file,
                        'relative_path': relative_path,
                        'directory': directory,
                        'namespace_or_package': namespace_or_package,
                        'search_filename': search_filename
                    }
                    
                    logger.info(f"Uploaded {file_path} (relative: {relative_path}) to Azure client")
        
        logger.info(f"Uploaded {len(uploaded_files)} files successfully")
        logger.info(f"File structure preserved for {len(file_metadata_map)} files")
        return uploaded_files, file_metadata_map, temp_files
    
    @staticmethod
    def _extract_namespace_from_path(relative_path: str, language: str) -> str:
        """
        Extract namespace or package from file path.
        
        Args:
            relative_path: Relative path of the file
            language: 'java' or 'csharp'
            
        Returns:
            Extracted namespace/package string
        """
        if language == 'java':
            # For Java, extract from src/main/java/ onwards
            if 'src/main/java/' in relative_path or 'src\\main\\java\\' in relative_path:
                # Split and get package path
                parts = relative_path.replace('\\', '/').split('src/main/java/')
                if len(parts) > 1:
                    package_path = parts[1].rsplit('/', 1)[0]  # Remove filename
                    return package_path.replace('/', '.')
            return ""
        else:  # csharp
            # For C#, extract from directory structure
            directory = os.path.dirname(relative_path)
            if directory:
                # Convert path to namespace (e.g., Functions/Handlers -> Functions.Handlers)
                return directory.replace('\\', '.').replace('/', '.')
            return ""
    
    @staticmethod
    async def create_vector_store(client, uploaded_files: List[str]) -> str:
        """
        Create a vector store with the uploaded files.
        
        Args:
            client: Azure AI client
            uploaded_files: List of uploaded file IDs
            
        Returns:
            Vector store ID
        """
        logger.info("Creating vector store")
        vector_store = await client.agents.create_vector_store_and_poll(
            file_ids=[],
            name="vector_store_tbd"
        )
        
        logger.info(f"Vector store created with ID: {vector_store.id}")
        logger.info("Adding files to vector store")
        
        vector_store_file_batch = await client.agents.create_vector_store_file_batch_and_poll(
            vector_store_id=vector_store.id,
            file_ids=uploaded_files
        )
        
        logger.info("Files added to vector store successfully")
        return vector_store.id
    
    @staticmethod
    async def create_agent(
        client,
        vector_store_id: str,
        kernel: Kernel
    ) -> Tuple[AzureAIAgent, AzureAIAgentThread]:
        """
        Create an Azure AI Agent with the specified configuration.
        
        Args:
            client: Azure AI client
            vector_store_id: ID of the vector store
            kernel: Semantic Kernel instance
            
        Returns:
            Tuple of (agent, thread)
        """
        logger.info("Creating Azure AI Agent")
        
        file_search = FileSearchTool(vector_store_ids=[vector_store_id])
        
        agent_definition = await client.agents.create_agent(
            model="gpt-4.1",
            tools=file_search.definitions,
            tool_resources=file_search.resources
        )
        
        agent = AzureAIAgent(client=client, definition=agent_definition, kernel=kernel)
        agent.polling_options.run_polling_timeout = timedelta(minutes=40)
        
        logger.info("Agent created successfully")
        logger.info("Creating agent thread")
        
        thread = AzureAIAgentThread(client=client)
        await thread.create()
        
        logger.info("Agent thread created successfully")
        return agent, thread
    
    @staticmethod
    def cleanup_temp_files(temp_files: List[str]):
        """
        Clean up temporary files created during processing.
        
        Args:
            temp_files: List of temporary file paths to remove
        """
        if not temp_files:
            return
        
        logger.info(f"Cleaning up {len(temp_files)} temporary files")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
