using Docker.DotNet;
using Docker.DotNet.Models;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Qdrant.Client;
using Spectre.Console;

namespace MarioTiscareno.AgentSample;

public static class Program
{
    private static string? qdrantContainerId;

    private static DockerClient? dockerClient;

    private static string openAIApiKey = "REPLACE";

    private static readonly string[] models = [
        "gpt-4o-mini",
        "gpt-4o",
        "o3-mini",
        "o1-mini",
        "o1",
    ];

    public static async Task Main()
    {
        if (openAIApiKey == "REPLACE")
        {
            AnsiConsole.MarkupLine("Please enter your Open AI API key [link green](https://platform.openai.com/settings/organization/api-keys)[/]");
            openAIApiKey = AnsiConsole.Prompt(
                new TextPrompt<string>(string.Empty));
        }

        var model = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("What [green]model[/] should I use?")
                .PageSize(10)
                .AddChoices(models));

        AnsiConsole.MarkupLine($"Using model: [green]{model}[/]");

        // Create a kernel with Azure OpenAI chat completion
        var builder = Kernel.CreateBuilder().AddOpenAIChatCompletion(model, openAIApiKey);

        if (AnsiConsole.Prompt(new TextPrompt<bool>("Do you want to enable [green]tracing[/]? This can help visualize Semantic Kernel's internal steps.")
            .AddChoice(false)
            .AddChoice(true)
            .DefaultValue(false)
            .WithConverter(choice => choice ? "y" : "n")))
        {
            builder.Services.AddLogging(services => services.AddConsole().SetMinimumLevel(LogLevel.Trace));
        }

        await InitializeEnvironmentAsync();

        // Build the kernel
        Kernel kernel = builder.Build();
        var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

        kernel.Plugins.AddFromObject(new WebPageReaderPlugin(openAIApiKey));

        // Enable planning
        OpenAIPromptExecutionSettings openAIPromptExecutionSettings = new()
        {
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
        };

        // Create a history store the conversation
        var history = new ChatHistory();

        var greeting = "Hello! I'm a web powered AI assistant. I can browse web pages and answer questions based on their content. Just give me a URL and a question.\n" +
            "E.g: \"Based on the contents found on https://www.gutenberg.org/cache/epub/74/pg74.txt who were Tom's best friends?\"";

        history.AddAssistantMessage(greeting);
        AnsiConsole.MarkupLine($"Assistant > [green]{greeting}[/]");

        // Initiate a back-and-forth chat
        string? userInput;
        do
        {
            // Collect user input
            userInput = AnsiConsole.Prompt(new TextPrompt<string>("User > "));

            if (userInput == "exit" || userInput == "quit")
            {
                break;
            }

            // Add user input
            history.AddUserMessage(userInput);

            // Get the response from the AI
            var result = await chatCompletionService.GetChatMessageContentAsync(
                history,
                executionSettings: openAIPromptExecutionSettings,
                kernel: kernel);

            // Print the results
            AnsiConsole.MarkupLine("Assistant > [green]" + result + "[/]");

            // Add the message from the agent to the chat history
            history.AddMessage(result.Role, result.Content ?? string.Empty);
        } while (userInput is not null);

        await DisposeEnvironmentAsync();
    }

    private static async Task InitializeEnvironmentAsync()
    {
        if (qdrantContainerId == null)
        {
            // Connect to docker and start the docker container.
            using var dockerClientConfiguration = new DockerClientConfiguration();
            dockerClient = dockerClientConfiguration.CreateClient();
            qdrantContainerId = await SetupQdrantContainerAsync(dockerClient);

            // Delay until the Qdrant server is ready.
            var qdrantClient = new QdrantClient("localhost");
            var succeeded = false;
            var attemptCount = 0;
            while (!succeeded && attemptCount++ < 10)
            {
                try
                {
                    await qdrantClient.ListCollectionsAsync();
                    succeeded = true;
                }
                catch (Exception)
                {
                    await Task.Delay(1000);
                }
            }
        }
    }

    private static async Task<string> SetupQdrantContainerAsync(DockerClient client)
    {
        await client.Images.CreateImageAsync(
            new ImagesCreateParameters
            {
                FromImage = "qdrant/qdrant",
                Tag = "latest",
            },
            null,
            new Progress<JSONMessage>());

        var containers = await client.Containers.ListContainersAsync(new ContainersListParameters
        {
            Filters = new Dictionary<string, IDictionary<string, bool>>
        {
            {
                "name", new Dictionary<string, bool>
                {
                    { "qdrant", true }
                }
            }
        }
        });

        var qdrantContainer = containers.FirstOrDefault();

        if (qdrantContainer?.State == "running")
        {
            return qdrantContainer.ID;
        }

        if (qdrantContainer is not null && qdrantContainer.State != "running")
        {
            await DeleteContainerAsync(client, qdrantContainer!.ID);
        }

        var container = await client.Containers.CreateContainerAsync(new CreateContainerParameters()
        {
            Image = "qdrant/qdrant",
            Name = "qdrant",
            HostConfig = new HostConfig()
            {
                PortBindings = new Dictionary<string, IList<PortBinding>>
            {
                {"6333", new List<PortBinding> {new() {HostPort = "6333" } }},
                {"6334", new List<PortBinding> {new() {HostPort = "6334" } }}
            },
                PublishAllPorts = true
            },
            ExposedPorts = new Dictionary<string, EmptyStruct>
        {
            { "6333", default },
            { "6334", default }
        },
        });

        await client.Containers.StartContainerAsync(
            container.ID,
            new ContainerStartParameters());

        return container.ID;
    }

    private static async Task DisposeEnvironmentAsync()
    {
        if (dockerClient != null && qdrantContainerId != null)
        {
            // Delete docker container.
            await DeleteContainerAsync(dockerClient, qdrantContainerId);
        }
    }

    private static async Task DeleteContainerAsync(DockerClient client, string containerId)
    {
        await client.Containers.StopContainerAsync(containerId, new ContainerStopParameters());
        await client.Containers.RemoveContainerAsync(containerId, new ContainerRemoveParameters());
    }
}