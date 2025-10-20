namespace RAG_LLM_INTEGRATION.Services
{
    public interface ILangChainService
    {
        Task<List<string>> GetChunkedDocument(IFormFile file);
        Task<bool> ChunkDocument(IFormFile file);
    }
}
