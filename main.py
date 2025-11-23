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

input_folder = "initial_codebase"


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

prompt = """
Analyze the provided Java AWS Lambda codebase. Your goal is to transform this into a deployment-ready Azure Function App project using Java (Java 17 or 21).

You must generate ALL necessary files for a complete Azure Function project, including:
1.  **Java Source Code**: Convert Lambda handlers to Azure Function classes.
    *   **One-to-One Mapping**: For each AWS Lambda handler class found (e.g., `HandlerSQS`), generate EXACTLY ONE corresponding Azure Function class (e.g., `HandlerSQSFunction`). Do NOT create multiple variations or duplicate files for the same handler.
    *   **Logic Preservation**: Analyze the logic inside the `handleRequest` method carefully.
        *   If the Lambda processes a batch of events (e.g., looping through `SQSEvent.getRecords()`), your Azure Function (which typically triggers once per message for Queue triggers) should implement the *inner logic* of that loop for the single message it receives.
        *   Do NOT simplify the logic. If the original code performs specific transformations, validations, or calls other methods, you MUST replicate that logic in the Azure Function.
        *   Preserve variable names and comments where applicable.
    *   Use appropriate triggers (e.g., `@HttpTrigger`, `@CosmosDBTrigger`, `@QueueTrigger`).
    *   Ensure correct package structure (e.g., `com.example.functions`).
2.  **Configuration Files**:
    *   `host.json`: Standard configuration for the function host.
    *   `local.settings.json`: For local development settings.
3.  **Build Configuration**:
    *   `pom.xml`: A complete Maven project file including `azure-functions-maven-plugin` and necessary dependencies.

Use the `file_writer` tool to write each file to the local file system. Ensure you use a standard Maven project structure:
*   `src/main/java/...` for source code.
*   `pom.xml`, `host.json`, `local.settings.json` in the root of the output directory.

Do not just output code snippets; write the full file contents.
"""


async def main():
    try:
        async with (
            DefaultAzureCredential() as credential,
            AzureAIAgent.create_client(
                credential = credential
            ) as client,
        ) :
            # Read pom.xml and README.md content
            extra_context = ""
            try:
                with open(os.path.join(input_folder, "pom.xml"), "r") as f:
                    extra_context += f"\n\nHere is the content of pom.xml:\n{f.read()}"
            except FileNotFoundError:
                pass
                
            try:
                with open(os.path.join(input_folder, "README.md"), "r") as f:
                    extra_context += f"\n\nHere is the content of README.md:\n{f.read()}"
            except FileNotFoundError:
                pass

            full_prompt = prompt + extra_context

            # Vector Store Creation
            uploaded_files = []
            file_id_map = {} # Map filename to file_id
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path.endswith((".java")):
                        _file = await client.agents.upload_file_and_poll(file_path = file_path, purpose = FilePurpose.AGENTS)
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
                # 1. Generate Project Configuration (pom.xml, host.json, local.settings.json)
                print("Generating project configuration...")
                config_prompt = prompt + "\n\nFirst, generate the project configuration files: `pom.xml`, `host.json`, and `local.settings.json`. Do not generate Java files yet."
                async for response in agent.invoke(messages = config_prompt, thread = thread):
                    print(response)

                # 2. Process each Java file individually
                for filename, file_id in file_id_map.items():
                    print(f"Processing {filename}...")
                    base_name = filename.replace(".java", "")
                    file_prompt = f"""
                    Focus ONLY on converting the file: {filename}.
                    1. Read the content of {filename}.
                    2. Generate the corresponding Azure Function Java class.
                       - Class Name: {base_name}Function (e.g., if original is HandlerSQS, new is HandlerSQSFunction).
                       - Package: com.example.functions
                       - Trigger: Use the appropriate Azure Function trigger.
                       - Logic: PRESERVE the exact business logic. If it's a batch loop, implement the inner loop logic for the single Azure message. Do not use placeholders. Replicate the code.
                    3. Use the file_writer tool to save it to `src/main/java/com/example/functions/{base_name}Function.java`.
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
            
    finally:
        print("Execution finished.")
if __name__ == "__main__":
    asyncio.run(main())
