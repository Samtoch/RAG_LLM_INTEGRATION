# LLM Optimization with RAG using Qdrant and OpenAI in .Net 8 API
This is a .Net 8 API Project of RAG implementation using Qdrant database, OpenAI APIs and Embedding Models. 
- Qdrant is used as the vector database to store and retrieve embeddings.
- OpenAI APIs are used to generate embeddings and perform semantic search.
- NLog is used for logging.
- Swagger is used for API documentation.
- The project is structured to follow best practices for .Net API development.
- The project includes error handling and logging for better maintainability.
- The project is designed to be scalable and can handle large volumes of data.
- The project is documented with clear instructions on how to set up and use the API.
- The project includes a Postman collection for easy testing of the API endpoints.
- The project supports both Ollama (local LLM) and OpenAI services for flexibility.
- The project includes a modular architecture to allow easy integration of additional services in the future.
- The project is open source and contributions are welcome.

## Prerequisites
- .Net 8 SDK
- Ollama installed and running locally OR access to OpenAI services
- OpenAI API Key (if using OpenAI API)
- Qdrant API Key
- Visual Studio or any other IDE for .Net development
- Postman or any other API testing tool
- Git
- Swagger (for API documentation)
- NLog (for logging)

## Getting Started
### Qdrant Setup
- Sign up to https://login.cloud.qdrant.io/ and create an account.
- Follow the Qdrant documentation to set up a Qdrant instance.
- Create a collection in Qdrant to store your embeddings.
- Obtain your Qdrant API key and endpoint URL.
- Install Qdrant client library for .Net.
- Configure the Qdrant client in your .Net project with your API key and endpoint URL.
- Test the connection to Qdrant to ensure it's working correctly.

- ### OpenAI Setup
- Sign up for an OpenAI account and obtain your API key.
- Install OpenAI client library for .Net.
- Configure the OpenAI client in your .Net project with your API key.
- Test the connection to OpenAI to ensure it's working correctly.

- ### Project Setup
- Fork or clone this repository.
- Clone the repository
- Navigate to the project directory
- Restore the NuGet packages
- Update the appSettings with your own qDrant and OpenAI credentials
- Build and run the project

## Configuration
- Update the `appsettings.json` file with your OpenAI and Qdrant API keys.
- Configure logging settings as needed.