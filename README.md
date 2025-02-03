# Agent Sample

This project demonstrates how to build a console-based AI agent using Semantic Kernel. The agent can retrieve context from web pages and answer questions based on the content.

## Features

- Retrieves context from web pages and stores embeddings to local Qdrant instance
- Answers questions based on the retrieved context
- Uses Semantic Kernel + Open AI for natural language processing
- Tracing through command line prompt for easy debugging

## Usage

1. Clone this repository

   ```
   git clone https://github.com/MarioTiscareno/agent-sample
   ```

2. Go into the console project directory

   ```
   cd agent-sample/MarioTiscareno.AgentSample
   ```

3. Run

   ```
   dotnet run
   ```

4. Follow the command line instructions

## Requirements

- .NET SDK 8.0 or later
- Open AI API key
- Docker to host a local Qdrant instance

## License

This project is licensed under the MIT License.
