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

        [Route("openai/chat")]
        [HttpGet]
        public async Task<ActionResult> ChatOpenAI(string prompt)
        {
            var response = await _searchService.GetOpenAIGpt4oMiniAnswer(prompt);
            return Ok(response);
        }

        [Route("openai/embeddings")]
        [HttpGet]
        public async Task<ActionResult> OpenAIEbeddings(string inputText)
        {
            var response = await _searchService.GetOpenAIEmbedding(inputText);
            return Ok(response);
        }

        [Route("chat/deepseek")]
        [HttpGet]
        public async Task<ActionResult> ChatDeepseek(string prompt)
        {
            var response = await _searchService.GetDeepSeekAnswer(prompt);
            return Ok(response);
        }

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
            //float[] questionEmbedding = await _searchService.GetOllamaEmbedding(request.InputText, model);
            float[] questionEmbedding = await _searchService.GetOpenAIEmbedding(request.InputText);

            var searchResults = await _searchService.SearchTopMatches(request.CollectionName, questionEmbedding, topK: 1);

            // Combine all texts into a single context
            var context = string.Join("\n", searchResults.Select(r => r.Name));

            string fullPrompt = $"Context:\n{context}\n\nQuestion: {request.InputText}\nAnswer:";

            //string answer = await _searchService.GetLlama3Answer(fullPrompt);
            string answer = await _searchService.GetOpenAIGpt4oMiniAnswer(fullPrompt);

            return Ok(new { Answer = answer });
        }

    }
}
