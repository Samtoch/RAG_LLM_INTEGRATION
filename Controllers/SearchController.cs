using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using RAG_LLM_INTEGRATION.Model;
using RAG_LLM_INTEGRATION.Services;

namespace RAG_LLM_INTEGRATION.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class SearchController : ControllerBase
    {
        private readonly ISearchService _searchService;

        public SearchController(ISearchService searchService)
        {
            _searchService = searchService;
        }

        //[Route("search")]
        //[HttpGet]
        //public async Task<ActionResult> SEmanticSearch(string inputText)
        //{
        //    var response = await _searchService.SemanticSearch(inputText);
        //    return Ok(response);
        //}

        [HttpPost("UploadCollection")]
        public async Task<IActionResult> UploadEmbeddings([FromBody] EmbeddingRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.CollectionName) || request.Entries == null || !request.Entries.Any())
                return BadRequest("Collection name and entries are required.");

            bool result = await _searchService.CreateEmbedingsOfAnimalsList(request.CollectionName, request.Entries);

            if (result)
                return Ok("Embeddings uploaded successfully.");
            else
                return StatusCode(500, "Failed to upload some or all embeddings.");
        }

        [HttpPost("SemanticSearch")]
        public async Task<IActionResult> SearchSimilar([FromBody] SearchRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.CollectionName) || string.IsNullOrWhiteSpace(request.InputText))
                return BadRequest("Collection name and input text required.");

            try
            {
                var matches = await _searchService.SemanticSearch(request.CollectionName, request.InputText);
                return Ok(matches);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Search failed: {ex.Message}");
            }
        }

        [HttpPost("rag/answer")]
        public async Task<IActionResult> GenerateAnswer([FromBody] SearchRequest request)
        {
            string model = "all-minilm";
            float[] questionEmbedding = await _searchService.GetOllamaEmbedding(request.InputText, model);

            var searchResults = await _searchService.SearchTopMatches(request.CollectionName, questionEmbedding, topK: 3);

            // Combine all texts into a single context
            var context = string.Join("\n", searchResults.Select(r => r.Name));

            string fullPrompt = $"Context:\n{context}\n\nQuestion: {request.InputText}\nAnswer:";

            string answer = await _searchService.GetLlama3Answer(fullPrompt);

            return Ok(new { Answer = answer });
        }

    }
}
