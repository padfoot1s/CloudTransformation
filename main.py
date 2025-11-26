import os

import asyncio
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread
from azure.identity.aio import DefaultAzureCredential
from azure.ai.agents.models import FileSearchTool
from azure.ai.projects.models import FilePurpose, FileSearchTool, CodeInterpreterTool 
from datetime import timedelta
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from pathlib import Path
load_dotenv()

input_folder = "initial_codebase_cs"


class FileWriter:
    def __init__(self):
        self._directory = os.path.join(Path(__file__).parents[0], "modified_files")

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
        return str(file_path)

def get_java_prompt():
    return """
Analyze the provided Java AWS Lambda codebase. Your goal is to transform this into a deployment-ready Azure Function App project using Java (Java 17 or 21).

IMPORTANT: Only convert production Lambda handler files. Do NOT convert test files.

**CRITICAL: CLOUD SERVICE MIGRATION & STRUCTURE**
1.  **Service Migration**: MIGRATE AWS service calls to Azure equivalents.
2.  **Architecture**: Use a clean, **multi-file architecture** with Dependency Injection (e.g., Spring Cloud Function or simple DI).
    *   **Models**: POCO classes for data structures (e.g., `SpotVmRequest`).
    *   **Services**: Interface-based services for business logic (e.g., `SpotVmService`).
    *   **Functions**: Azure Function classes that inject services.

**Service Mapping**:
*   **EC2** -> **Azure Virtual Machines** (Use `com.azure.resourcemanager:azure-resourcemanager-compute`)
*   **S3** -> **Azure Blob Storage** (Use `com.azure:azure-storage-blob`)
*   **DynamoDB** -> **Azure Cosmos DB** (Use `com.azure:azure-cosmos`)
*   **SQS** -> **Azure Queue Storage** (Use `com.azure:azure-storage-queue`)

**Concept Mapping**:
*   **AMI IDs** -> **Azure Image References**
*   **Instance Types** -> **Azure VM Sizes**
*   **Spot Instances** -> **Azure Spot VMs** (`Priority = VirtualMachinePriority.SPOT`)

**SDK Usage**:
*   Remove `aws-java-sdk-*` dependencies.
*   Add `com.azure:azure-identity`, `com.azure.resourcemanager:azure-resourcemanager`, etc.

You must generate ALL necessary files for a complete Azure Function project.
**Output Structure**:
*   `src/main/java/com/example/functions/{base_name}Function.java`: Entry point.
*   `src/main/java/com/example/services/I{base_name}Service.java`: Interface.
*   `src/main/java/com/example/services/{base_name}Service.java`: Implementation.
*   `src/main/java/com/example/models/{ModelName}.java`: Data models.
*   `pom.xml`: Complete Maven project file.
*   `host.json`, `local.settings.json`.

Use the `file_writer` tool to write each file to the local file system.
Do not just output code snippets; write the full file contents.
"""

def get_csharp_prompt():
    return """
Analyze the provided C# AWS Lambda codebase. Your goal is to transform this into a deployment-ready Azure Function App project using C# (.NET 6 or .NET 8).

IMPORTANT: Only convert production Lambda handler files. Do NOT convert test files.

**CRITICAL: CLOUD SERVICE MIGRATION & STRUCTURE**
1.  **Service Migration**: MIGRATE AWS service calls to Azure equivalents (EC2 -> Azure Compute, S3 -> Blob Storage).
2.  **Architecture**: Use a clean, **multi-file architecture** with Dependency Injection.
    *   **Models**: POCO classes for data structures (e.g., `SpotVmRequest`).
    *   **Services**: Interface-based services for business logic (e.g., `ISpotVmService`, `SpotVmService`).
    *   **Functions**: Azure Function classes that inject services and handle triggers.
    *   **Program.cs**: Register services and Azure clients using `Microsoft.Extensions.DependencyInjection`.

**Service Mapping**:
*   **EC2** -> **Azure Virtual Machines** (Use `Azure.ResourceManager.Compute`)
*   **S3** -> **Azure Blob Storage** (Use `Azure.Storage.Blobs`)
*   **DynamoDB** -> **Azure Cosmos DB** (Use `Microsoft.Azure.Cosmos`)
*   **SQS** -> **Azure Queue Storage** (Use `Azure.Storage.Queues`)

**Concept Mapping**:
*   **AMI IDs** -> **Azure Image References**
*   **Instance Types** -> **Azure VM Sizes**
*   **Spot Instances** -> **Azure Spot VMs** (`Priority = VirtualMachinePriority.Spot`)

**SDK Usage**:
*   Remove `AWSSDK.*` packages.
*   Add `Azure.Identity`, `Azure.ResourceManager`, `Azure.ResourceManager.Compute`, etc.

You must generate ALL necessary files for a complete Azure Function project.
**Output Structure**:
*   `Functions/{base_name}Function.cs`: The Azure Function entry point.
*   `Services/I{base_name}Service.cs`: Service interface.
*   `Services/{base_name}Service.cs`: Service implementation (contains the migrated logic).
*   `Models/{ModelName}.cs`: Data models.
*   `Program.cs`: DI configuration.
*   `*.csproj`: Complete project file with Azure SDKs.
*   `host.json`, `local.settings.json`.

Use the `file_writer` tool to write each file to the local file system.
Do not just output code snippets; write the full file contents.
"""

def detect_language(folder_path):
    """Detect the primary language in the codebase by scanning for .java or .cs files."""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.java'):
                return 'java'
            elif file.endswith('.cs'):
                return 'csharp'
    return None


async def main():
    try:
        # Detect language
        language = detect_language(input_folder)
        if language is None:
            print("Error: No Java or C# files found in the input folder.")
            return
        
        print(f"Detected language: {language.upper()}")
        
        # Get language-specific prompt
        if language == 'java':
            prompt = get_java_prompt()
            file_extension = '.java'
            config_files = ['pom.xml', 'README.md']
            source_path_template = 'src/main/java/com/example/functions/{base_name}Function.java'
        else:  # csharp
            prompt = get_csharp_prompt()
            file_extension = '.cs'
            config_files = ['*.csproj', 'README.md']
            source_path_template = '{base_name}Function.cs'
        
        async with (
            DefaultAzureCredential() as credential,
            AzureAIAgent.create_client(
                credential = credential
            ) as client,
        ) :
            # Read configuration files and README.md content
            extra_context = ""
            
            # For C#, find .csproj files (skip test projects)
            if language == 'csharp':
                for root, dirs, files in os.walk(input_folder):
                    for file in files:
                        if file.endswith('.csproj'):
                            file_path = os.path.join(root, file)
                            # Skip test project files
                            if '/test/' in file_path.lower() or '/tests/' in file_path.lower():
                                print(f"Skipping test project: {file_path}")
                                continue
                            try:
                                with open(file_path, "r") as f:
                                    extra_context += f"\n\nHere is the content of {file}:\n{f.read()}"
                            except FileNotFoundError:
                                pass
            else:  # Java
                try:
                    with open(os.path.join(input_folder, "pom.xml"), "r") as f:
                        extra_context += f"\n\nHere is the content of pom.xml:\n{f.read()}"
                except FileNotFoundError:
                    pass
            
            # Read README.md for both languages
            try:
                with open(os.path.join(input_folder, "README.md"), "r") as f:
                    extra_context += f"\n\nHere is the content of README.md:\n{f.read()}"
            except FileNotFoundError:
                pass

            full_prompt = prompt + extra_context

            # Vector Store Creation
            uploaded_files = []
            file_id_map = {} # Map filename to file_id
            temp_files = []  # Track temporary files for cleanup
            
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path.endswith(file_extension):
                        # Skip test files
                        if '/test/' in file_path.lower() or '/tests/' in file_path.lower():
                            print(f"Skipping test file: {file_path}")
                            continue
                        
                        upload_path = file_path
                        
                        # Workaround: Azure doesn't support .cs extension, so create a .txt copy
                        if language == 'csharp':
                            temp_path = file_path + '.txt'
                            import shutil
                            shutil.copy2(file_path, temp_path)
                            upload_path = temp_path
                            temp_files.append(temp_path)
                            print(f"Created temporary file {temp_path} for upload")
                        
                        _file = await client.agents.upload_file_and_poll(file_path = upload_path, purpose = FilePurpose.AGENTS)
                        uploaded_files.append(_file.id)
                        file_id_map[file] = _file.id
                        print(f"Uploaded {file_path} to Azure client.")

            vector_store = await client.agents.create_vector_store_and_poll(file_ids = [], name = "vector_store_tbd")
            vector_store_file_batch = await client.agents.create_vector_store_file_batch_and_poll(
                vector_store_id = vector_store.id,
                file_ids = uploaded_files
            )
            
            file_search = FileSearchTool(vector_store_ids=[vector_store.id])
            kernel = Kernel()
            kernel.add_plugin(FileWriter())
            agent_definition = await client.agents.create_agent(
                model = "gpt-4.1",
                tools = file_search.definitions,
                tool_resources = file_search.resources
            )

            agent = AzureAIAgent(client = client, definition = agent_definition, kernel = kernel)
            agent.polling_options.run_polling_timeout = timedelta(minutes = 40)
            thread = AzureAIAgentThread(client=client)
            await thread.create()
            
            try:
                # 1. Generate Project Configuration
                print("Generating project configuration...")
                if language == 'java':
                    config_prompt = full_prompt + "\n\nFirst, generate the project configuration files: `pom.xml`, `host.json`, and `local.settings.json`. Do not generate Java files yet."
                else:  # csharp
                    config_prompt = full_prompt + "\n\nFirst, generate the project configuration files: `*.csproj`, `host.json`, and `local.settings.json`. Do not generate C# files yet."
                
                async for response in agent.invoke(messages = config_prompt, thread = thread):
                    print(response)

                # 2. Process each source file individually
                for filename, file_id in file_id_map.items():
                    print(f"Processing {filename}...")
                    base_name = filename.replace(file_extension, "")
                    
                    if language == 'java':
                        file_prompt = f"""
                        Focus ONLY on converting the file: {filename}.
                        1. Read the content of {filename}.
                        2. Generate a **multi-file** Azure Function solution with Dependency Injection.
                           - **src/main/java/com/example/functions/{base_name}Function.java**: Azure Function entry point. Inject `{base_name}Service`.
                           - **src/main/java/com/example/services/I{base_name}Service.java**: Service interface.
                           - **src/main/java/com/example/services/{base_name}Service.java**: Implementation. **MIGRATE** AWS logic here using Azure SDKs.
                           - **src/main/java/com/example/models/**: Create separate files for data models.
                           - **Logic Migration**:
                               - **MIGRATE** AWS service calls to Azure equivalents.
                               - **REPLACE** AWS SDKs with Azure SDKs.
                               - **MAP** concepts: Spot Instances -> Azure Spot VMs.
                        3. Use the file_writer tool to save ALL generated files to their respective paths.
                        """
                    else:  # csharp
                        # The file was uploaded as .txt, so we must ask the agent to read that specific filename
                        search_filename = filename + ".txt"
                        file_prompt = f"""
                        Focus ONLY on converting the file: {filename}.
                        1. Read the content of {search_filename} (which contains the code for {filename}).
                        2. Analyze the original namespace and class structure in {filename}.
                        3. Generate a **multi-file** Azure Function solution with Dependency Injection.
                           - **Functions/{base_name}Function.cs**: The Azure Function entry point. Inject `I{base_name}Service`.
                           - **Services/I{base_name}Service.cs**: Interface defining the business logic methods.
                           - **Services/{base_name}Service.cs**: Implementation of the service. **MIGRATE** AWS logic here using Azure SDKs.
                           - **Models/**: Create separate files for any data models (e.g., request/response objects).
                           - **Program.cs**: Register `I{base_name}Service` and Azure clients (e.g., `ArmClient`) in the DI container.
                           - **Namespace**: PRESERVE the exact namespace from {filename}.
                           - **Logic Migration**:
                               - **MIGRATE** AWS service calls to Azure equivalents (e.g., EC2 -> Azure Compute).
                               - **REPLACE** AWS SDKs with Azure SDKs.
                               - **MAP** concepts: Spot Instances -> Azure Spot VMs.
                        4. Use the file_writer tool to save ALL generated files to their respective paths.
                        """
                    
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            async for response in agent.invoke(messages = file_prompt, thread = thread):
                                print(response)
                            break
                        except Exception as e:
                            if "Rate limit is exceeded" in str(e):
                                wait_time = (attempt + 1) * 20
                                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                print(f"Error processing {filename}: {e}")
                                break
                                
            finally:
                print("Closing thread...")
                await thread.delete()
                await client.close()
                
                # Cleanup temporary files
                if temp_files:
                    import os as os_module
                    for temp_file in temp_files:
                        try:
                            os_module.remove(temp_file)
                            print(f"Cleaned up temporary file: {temp_file}")
                        except Exception as e:
                            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
            
    finally:
        print("Execution finished.")
if __name__ == "__main__":
    asyncio.run(main())
