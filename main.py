import os
import asyncio
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import AzureAIAgent
from azure.identity.aio import DefaultAzureCredential
from datetime import timedelta
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from pathlib import Path

from src.agents.AgentFactory import AgentFactory
from src.utils.logging_config import setup_logging, get_logger
from src.plugins.file_writer import FileWriter

load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)

input_folder = "initial_codebase_java"


async def main():
    """Main entry point for the AWS to Azure transformation process."""
    try:
        logger.info("Starting AWS to Azure transformation process")
        
        # Detect language using AgentFactory
        language = AgentFactory.detect_language(input_folder)
        if language is None:
            logger.error("No Java or C# files found in the input folder")
            return
        
        # Get language-specific configuration
        file_extension = '.java' if language == 'java' else '.cs'
        
        # Get prompt and extra context
        prompt = AgentFactory.get_prompt(language)
        extra_context = AgentFactory.read_extra_context(input_folder, language)
        full_prompt = prompt + extra_context
        
        async with (
            DefaultAzureCredential() as credential,
            AzureAIAgent.create_client(credential=credential) as client,
        ):
            logger.info("Azure client initialized successfully")
            
            # Upload files using AgentFactory
            uploaded_files, file_id_map, temp_files = await AgentFactory.upload_files(
                client, input_folder, language
            )
            
            # Create vector store using AgentFactory
            vector_store_id = await AgentFactory.create_vector_store(client, uploaded_files)
            
            # Create kernel and add FileWriter plugin
            kernel = Kernel()
            kernel.add_plugin(FileWriter())
            logger.info("Kernel initialized with FileWriter plugin")
            
            # Create agent and thread using AgentFactory
            agent, thread = await AgentFactory.create_agent(client, vector_store_id, kernel)
            
            try:
                # 1. Generate Project Configuration
                logger.info("Generating project configuration files")
                if language == 'java':
                    config_prompt = full_prompt + "\n\nFirst, generate the project configuration files: `pom.xml`, `host.json`, and `local.settings.json`. Do not generate Java files yet."
                else:  # csharp
                    config_prompt = full_prompt + "\n\nFirst, generate the project configuration files: `*.csproj`, `host.json`, and `local.settings.json`. Do not generate C# files yet."
                
                async for response in agent.invoke(messages=config_prompt, thread=thread):
                    logger.info(f"Agent response: {response}")

                # 2. Process each source file individually
                for relative_path, metadata in file_id_map.items():
                    logger.info(f"Processing file: {metadata['filename']} (path: {relative_path})")
                    
                    filename = metadata['filename']
                    search_filename = metadata['search_filename']
                    directory = metadata['directory']
                    namespace_or_package = metadata['namespace_or_package']
                    base_name = filename.replace(file_extension, "")
                    
                    if language == 'java':
                        file_prompt = f"""
                        **CONVERT THIS FILE IN-PLACE**
                        
                        File to convert: {search_filename}
                        Original path: {relative_path}
                        Original package: {namespace_or_package if namespace_or_package else 'com.example'}
                        
                        Instructions:
                        1.  **Read and Analyze**: Read the content of {search_filename}. Determine if it is a Lambda Handler, a Test, or a Utility/Model class.
                        2.  **Action based on Type**:
                            *   **Handler**: Convert to Azure Function (@FunctionName). Replacements: Context->ExecutionContext, APIGatewayProxyRequestEvent->HttpRequestMessage, etc.
                            *   **Test**: Keep as Test. Update mock/assertion libraries to match Azure SDKs.
                            *   **Utility/Other**: Keep logic. Replace AWS SDK calls with Azure SDK equivalents.
                        3.  **Path & Preservation**: Save to the EXACT SAME path `{relative_path}` with the same class name and package.
                        
                        Use the file_writer tool to save the converted file to: `{relative_path}`
                        """
                    else:  # csharp
                        file_prompt = f"""
                        **CONVERT THIS FILE IN-PLACE**
                        
                        File to convert: {search_filename}
                        Original path: {relative_path}
                        Original namespace: {namespace_or_package if namespace_or_package else 'MyCompany.AzureFunctions'}
                        
                        Instructions:
                        1.  **Read and Analyze**: Read the content of {search_filename}. Determine if it is a Lambda Handler, a Test, or a Utility/Model class.
                        2.  **Action based on Type**:
                            *   **Handler**: Convert to Azure Function ([FunctionName]). Replacements: ILambdaContext->ExecutionContext, APIGatewayProxyRequest->HttpRequest, etc.
                            *   **Test**: Keep as Test. Update mock/assertion libraries to match Azure SDKs.
                            *   **Utility/Other**: Keep logic. Replace AWS SDK calls with Azure SDK equivalents.
                        3.  **Path & Preservation**: Save to the EXACT SAME path `{relative_path}` with the same class name and namespace.
                        
                        Use the file_writer tool to save the converted file to: `{relative_path}`
                        """
                    
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            async for response in agent.invoke(messages=file_prompt, thread=thread):
                                logger.info(f"Agent response: {response}")
                            break
                        except Exception as e:
                            if "Rate limit is exceeded" in str(e):
                                wait_time = (attempt + 1) * 20
                                logger.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"Error processing {filename}: {e}")
                                break
                                
            finally:
                logger.info("Closing thread and cleaning up resources")
                await thread.delete()
                await client.close()
                
                # Cleanup temporary files using AgentFactory
                AgentFactory.cleanup_temp_files(temp_files)
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
    finally:
        logger.info("Execution finished")


if __name__ == "__main__":
    asyncio.run(main())
