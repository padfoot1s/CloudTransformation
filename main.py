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
                        1. Read the content of {search_filename}
                        2. Convert the AWS Lambda handler to an Azure Function
                        3. **CRITICAL**: Save the converted file to the EXACT SAME path: `{relative_path}`
                        4. Keep the same class name: `{base_name}`
                        5. Preserve the package: `{namespace_or_package if namespace_or_package else 'com.example'}`
                        6. Convert AWS SDK calls to Azure SDK equivalents:
                           - EC2 -> Azure Virtual Machines
                           - S3 -> Azure Blob Storage
                           - DynamoDB -> Azure Cosmos DB
                           - SQS -> Azure Queue Storage
                        7. Update method signature to use Azure Function annotations:
                           - Add @FunctionName("{base_name}")
                           - Use @HttpTrigger for HTTP endpoints
                           - Replace Context with ExecutionContext
                        8. Keep ALL business logic in the same file
                        
                        Use the file_writer tool to save the converted file to: `{relative_path}`
                        """
                    else:  # csharp
                        file_prompt = f"""
                        **CONVERT THIS FILE IN-PLACE**
                        
                        File to convert: {search_filename}
                        Original path: {relative_path}
                        Original namespace: {namespace_or_package if namespace_or_package else 'MyCompany.AzureFunctions'}
                        
                        Instructions:
                        1. Read the content of {search_filename} (which contains the code for {filename})
                        2. Convert the AWS Lambda function to an Azure Function
                        3. **CRITICAL**: Save the converted file to the EXACT SAME path: `{relative_path}`
                        4. Keep the same class name: `{base_name}`
                        5. Preserve the namespace: `{namespace_or_package if namespace_or_package else 'MyCompany.AzureFunctions'}`
                        6. Convert AWS SDK calls to Azure SDK equivalents:
                           - EC2 -> Azure Virtual Machines (Azure.ResourceManager.Compute)
                           - S3 -> Azure Blob Storage (Azure.Storage.Blobs)
                           - DynamoDB -> Azure Cosmos DB (Microsoft.Azure.Cosmos)
                           - SQS -> Azure Queue Storage (Azure.Storage.Queues)
                        7. Update method signature to use Azure Function attributes:
                           - Add [FunctionName("{base_name}")]
                           - Use [HttpTrigger] for HTTP endpoints
                           - Replace ILambdaContext with ExecutionContext
                           - Replace APIGatewayProxyRequest with HttpRequest
                           - Replace APIGatewayProxyResponse with IActionResult
                        8. Keep ALL business logic in the same file
                        9. Add dependency injection via constructor if needed
                        
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
