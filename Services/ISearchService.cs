using RAG_LLM_INTEGRATION.Model;
using System.Threading.Tasks;

namespace RAG_LLM_INTEGRATION.Services
{
    public interface ISearchService
    {
        Task<(bool Success, string Message)> CreateEmbedingsForCollection(string collectionName, List<string> entries);
        Task<List<SearchMatch>> SemanticSearch(string collectionName, string inputText);

        Task<string> GetLlama3Answer(string prompt);
        Task<string> GetDeepSeekAnswer(string prompt);
        Task<string> GetOpenAIGpt4oMiniAnswer(string prompt);

        Task<float[]> GetOllamaEmbedding(string text, string model);
        Task<float[]> GetOpenAIEmbedding(string text);

        Task<List<SearchMatch>> SearchTopMatches(string collectionName, float[] vector, int topK);
    }
}
