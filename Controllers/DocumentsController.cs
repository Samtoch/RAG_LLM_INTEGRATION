using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using RAG_LLM_INTEGRATION.Model;
using RAG_LLM_INTEGRATION.Services;

namespace RAG_LLM_INTEGRATION.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class DocumentsController : ControllerBase
    {
        //private readonly ISearchService _searchService;
        private readonly ILangChainService _langChainService;
        private readonly ISearchService _searchService;
        public DocumentsController(ILangChainService langChainService, ISearchService searchService)
        {
            _langChainService = langChainService;
            _searchService = searchService;
        }

        /// <summary>
        /// Upload a PDF, chunk it and return the chunks
        /// </summary>
        [HttpPost("GetChunks")]
        public async Task<IActionResult> GetDocument(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("No file uploaded.");
            }

            try
            {
                var res = await _langChainService.GetChunkedDocument(file);
                return Ok(res);
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { message = "Error processing file", error = ex.Message });
            }
        }

        /// <summary>
        /// Upload a PDF, chunk it, embed, and store in Qdrant
        /// </summary>
        [HttpPost("upload")]
        public async Task<IActionResult> UploadDocument(IFormFile file, string collectionName)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("No file uploaded.");
            }

            try
            {
                bool res = await _langChainService.ChunkDocumentAndCreateEmbeddings(file, collectionName);
                if (!res)
                    return StatusCode(500, new { message = "Failed to process and store embeddings" });
                return Ok(new { message = "File processed and embeddings stored successfully" });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { message = "Error processing file", error = ex.Message });
            }
        }

        /// <summary>
        /// Creates a , chunk it, embed, and store in Qdrant
        /// </summary>
        [HttpPost("CreateAndUploadCollection")]
        public async Task<IActionResult> UploadEmbeddings([FromBody] EmbeddingRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.CollectionName) || request.Entries == null || !request.Entries.Any())
                return BadRequest("Collection name and entries are required.");

            var (result, response) = await _searchService.CreateEmbedingsForCollection(request.CollectionName, request.Entries);

            if (result)
                return Ok("Embeddings uploaded successfully.");
            else
                return StatusCode(500, $"Something went wrong: {response}");
        }
    }
}
