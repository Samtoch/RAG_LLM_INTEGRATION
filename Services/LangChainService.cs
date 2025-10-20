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

                var textSplitter = new TokenTextSplitter(
                    chunkSize: 200,    // number of tokens per chunk
                    chunkOverlap: 50   // overlap between chunks
                );
                List<Document> chunks = textSplitter.SplitDocuments(documents).ToList();

                //List<Document> chunks = new List<Document>();
                //// Split into chunks (~500 chars, 100 overlap)
                //var textSplitter = new RecursiveCharacterTextSplitter(chunkSize: 500, chunkOverlap: 100);
                //List<Document> chunks = textSplitter.SplitDocuments(documents).ToList();

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

        public async Task<bool> ChunkDocument(IFormFile file)
        {
            try
            {
                string collectionName = "documents_collection";

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
