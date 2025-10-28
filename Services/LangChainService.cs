using LangChain.DocumentLoaders;
using LangChain.Extensions;
using LangChain.Schema;
using LangChain.Splitters;
using LangChain.Splitters.Text;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using OpenAI.Embeddings;
using Qdrant.Client;
using RAG_LLM_INTEGRATION.Helpers;
using Build5Nines.SharpVector;
using Build5Nines.SharpVector.Data;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.Extensions.AI;
using SemanticChunkerNET;
using UglyToad.PdfPig;


namespace RAG_LLM_INTEGRATION.Services
{
    public class LangChainService : ILangChainService
    {
        private readonly ILogger<LangChainService> _logger;

        private readonly ISearchService _searchService;
        private readonly IQdrantService _qdrantService;
        private readonly IConfiguration _configuration;
        private string? _qdrantUrl;
        private string? _qdrantKey;
        private string? _openAIAPIKey;

        public LangChainService(IConfiguration configuration, ILogger<LangChainService> logger, ISearchService searchService, IQdrantService qdrantService)
        {
            _logger = logger;
            _configuration = configuration;
            _qdrantUrl = _configuration["QdrantUrl"];
            _qdrantKey = _configuration["QdrantApiKey"];
            _openAIAPIKey = _configuration["OpenAIApiKey"];
            _searchService = searchService;
            _qdrantService = qdrantService;

        }

        public async Task<List<string>> GetChunkedDocument_(IFormFile file)
        {
            try
            {
                string collectionName = "documents_collection";

                // Save uploaded PDF to temp path
                string tempPath = Path.GetTempFileName();
                using (var stream = File.Create(tempPath))
                {
                    await file.CopyToAsync(stream);
                }

                // Extract text from PDF
                string fullText = ExtractTextFromPdf(tempPath);

                // Setup Semantic Kernel
                var builder = Kernel.CreateBuilder();

#pragma warning disable SKEXP0010
                builder.Services.AddOpenAIEmbeddingGenerator("text-embedding-ada-002", _openAIAPIKey);
#pragma warning restore SKEXP0010

                var kernel = builder.Build();

                var embeddingGenerator = kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

                // Chunk the document
                var chunker = new SemanticChunker(embeddingGenerator, tokenLimit: 512);
                var chunks = await chunker.CreateChunksAsync(fullText);

                // Convert chunks to list of entries
                var entries = chunks.Select(c => c.Text).ToList();

                // Store embeddings in Qdrant
                var (success, message) = await _searchService.CreateEmbedingsForCollection(collectionName, entries);
                if (!success)
                {
                    Console.WriteLine($"Embedding creation failed: {message}");
                    return null;
                }

                return entries;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                return null;
            }
        }

        private string ExtractTextFromPdf(string path)
        {
            using var pdf = PdfDocument.Open(path);
            var text = "";
            foreach (var page in pdf.GetPages())
            {
                text += page.Text + "\n";
            }
            return text;
        }

        private List<string> SplitIntoSentences(string text)
        {
            var sentences = new List<string>();
            var rawSentences = text.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (var sentence in rawSentences)
            {
                var trimmed = sentence.Trim();
                if (!string.IsNullOrWhiteSpace(trimmed))
                    sentences.Add(trimmed + ".");
            }
            return sentences;
        }

        public async Task<List<string>> GetChunkedDocument(IFormFile file)
        {
            try
            {
                string tempPath = Path.GetTempFileName();
                using (var stream = File.Create(tempPath))
                {
                    await file.CopyToAsync(stream);
                }

                var loader = new PdfDocumentLoader(tempPath);
                List<Document> documents = await loader.LoadAsync();

                // Split into chunks (~500 chars, 100 overlap)
                var textSplitter = new RecursiveCharacterTextSplitter(chunkSize: 500, chunkOverlap: 100);
                List<Document> chunks = textSplitter.SplitDocuments(documents).ToList();

                // 
                List<string> chunkTexts = chunks.Select(c => c.PageContent).ToList();

                return chunkTexts;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to process and chunk document");
                return null;
            }
        }

        public async Task<bool> ChunkDocumentAndCreateEmbeddings(IFormFile file, string collectionName)
        {
            try
            {
                //string collectionName = "documents_collection"; ChunkDocumentAndCreateEmbeddings

                // 1. Load the PDF from the uploaded file
                string tempPath = Path.GetTempFileName();
                using (var stream = File.Create(tempPath))
                {
                    await file.CopyToAsync(stream);
                }

                var loader = new PdfDocumentLoader(tempPath);
                List<Document> documents = await loader.LoadAsync();

                // 2. Split into chunks (~500 chars, 100 overlap)
                var textSplitter = new RecursiveCharacterTextSplitter(chunkSize: 500, chunkOverlap: 100);
                List<Document> chunks = textSplitter.SplitDocuments(documents).ToList();

                List<string> chunkTexts = chunks.Select(c => c.PageContent).ToList();

                // Ensure collection exists
                var (exists, msg) = await _qdrantService.CollectionExists(collectionName);
                if (!exists)
                {
                    await _qdrantService.CreateCollection(collectionName);
                }

                // 3. Embed chunks
                var embeddingModel = await _searchService.CreateEmbedingsForCollection(collectionName, chunkTexts);

                //await qdrant.UpsertAsync(collectionName, points);
                _logger.LogInformation("Successfully uploaded {Count} chunks into {Collection}", chunks.Count, collectionName);

                // Cleanup
                File.Delete(tempPath);

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to process and chunk document");
                return false;
            }
        }

    }
}
