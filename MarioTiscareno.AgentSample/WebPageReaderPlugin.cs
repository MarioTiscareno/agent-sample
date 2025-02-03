using Abot2.Core;
using HtmlAgilityPack;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Connectors.Qdrant;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Text;
using Qdrant.Client;
using System.ComponentModel;
using System.Text;
using static Qdrant.Client.Grpc.Conditions;

namespace MarioTiscareno.AgentSample;

public class WebPageReaderPlugin
{
    private readonly QdrantVectorStore vectorStore;

    private readonly QdrantClient qdrantClient;

    private readonly Task initializeTask;

    private IVectorStoreRecordCollection<ulong, WebPageChunk>? collection;

    private readonly OpenAITextEmbeddingGenerationService embeddingService;

    private readonly PageRequester pageRequester = new(new()
    {
        MaxPagesToCrawl = 10,
        MinCrawlDelayPerDomainMilliSeconds = 250,
    }, new WebContentExtractor());

    public WebPageReaderPlugin(string openAIKey)
    {
        qdrantClient = new QdrantClient("localhost");
        vectorStore = new QdrantVectorStore(qdrantClient);
        initializeTask = InitializeAsync();
        embeddingService = new OpenAITextEmbeddingGenerationService(
                "text-embedding-3-small",
                openAIKey);
    }

    private Task InitializeAsync()
    {
        collection = vectorStore.GetCollection<ulong, WebPageChunk>("web_pages");
        return collection.CreateCollectionIfNotExistsAsync();
    }

    [KernelFunction("read_web_page")]
    [Description("Reads a web page when given a url and it will try to return content that is relevant to an input question." +
        "The question is optional and can be omitted only if there is no question to answer, in such case the returned text will be the first paragraphs from that page.")]
    public async Task<string> ReadPageAsync(string url, string? optionalQuestion = null)
    {
        await initializeTask;

        var searchResult = await qdrantClient.QueryAsync("web_pages", filter: MatchText("Url", url), limit: 1);

        if (!searchResult.Any())
        {
            await AddWebPageToVectorStoreAsync(new Uri(url));
        }

        return await GetContentFromVectorStoreAsync(new Uri(url), optionalQuestion);
    }

    private async Task AddWebPageToVectorStoreAsync(Uri uri)
    {
        var crawledPage = await pageRequester.MakeRequestAsync(uri);

        var htmlDoc = new HtmlDocument();
        htmlDoc.LoadHtml(crawledPage.Content.Text);

        string plainText = ConvertToPlainText(htmlDoc.DocumentNode);

        var lines = TextChunker.SplitPlainTextLines(plainText, 50);
        var paragraphs = TextChunker.SplitPlainTextParagraphs(lines, 250);
        var urlString = crawledPage.Uri.ToString();

        var tasks = paragraphs.Select((p, i) => Task.Run(async () => new WebPageChunk
        {
            Key = Guid.NewGuid(),
            Url = urlString,
            Content = p,
            Chunk = i,
            DefinitionEmbedding = await embeddingService.GenerateEmbeddingAsync(p)
        }));

        var data = await Task.WhenAll(tasks);

        var upserTasks = data.Select(d => collection!.UpsertAsync(d));

        await Task.WhenAll(upserTasks);
    }

    private async Task<string> GetContentFromVectorStoreAsync(Uri uri, string? optionalQuestion = null)
    {
        if (optionalQuestion is null)
        {
            var condition = MatchText("Url", uri.ToString()) & Range("Chunk", new Qdrant.Client.Grpc.Range { Lt = 5.0 });
            var queryResult = await qdrantClient.QueryAsync("web_pages", filter: condition, limit: 5);

            return string.Join(" ", queryResult
                .OrderBy(r => r.Payload["Chunk"].IntegerValue)
                .Select(r => r.Payload["Content"].StringValue));
        }

        var searchVector = await embeddingService.GenerateEmbeddingAsync(optionalQuestion);
        var searchResult = await collection!.VectorizedSearchAsync(searchVector, new() { Top = 10, Filter = new VectorSearchFilter().EqualTo("Url", uri.ToString()) });
        var resultRecords = await searchResult.Results.ToListAsync();
        var relevantContent = string.Join(' ', resultRecords.Where(r => r.Score >= 0.5).Select(r => r.Record.Content));

        return relevantContent;
    }

    private static string ConvertToPlainText(HtmlNode node)
    {
        var sb = new StringBuilder();
        foreach (HtmlNode subnode in node.ChildNodes)
        {
            if (subnode.Name == "head"
                || subnode.Name == "script")
            {
                continue;
            }

            if (string.IsNullOrWhiteSpace(subnode.InnerText))
            {
                continue;
            }

            if (subnode.NodeType == HtmlNodeType.Text)
            {                // Append the text of the current node to the StringBuilder
                sb.Append(
                    string.Join(" ", subnode.InnerText
                        .Replace("\n", " ")
                        .Replace("\t", " ")
                        .Split(Array.Empty<char>(), StringSplitOptions.RemoveEmptyEntries)));

                sb.Append(' ');
            }
            else if (subnode.NodeType == HtmlNodeType.Element)
            {
                // Recursively convert the child nodes to plain text
                sb.Append(ConvertToPlainText(subnode));
            }
        }
        return sb.ToString();
    }
}

public class WebPageChunk
{
    [VectorStoreRecordKey]
    public Guid Key { get; set; }

    [VectorStoreRecordData]
    public required string Url { get; set; }

    [VectorStoreRecordData]
    public int Chunk { get; set; }

    [VectorStoreRecordData]
    public required string Content { get; set; }

    [VectorStoreRecordVector(1536)]
    public ReadOnlyMemory<float> DefinitionEmbedding { get; set; }
}