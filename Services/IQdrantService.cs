namespace RAG_LLM_INTEGRATION.Services
{
    public interface IQdrantService
    {
        Task CreateCollection(string collectionName);
        Task UploadVector(string collectionName, int id, float[] vector, string name);
        Task<float[]> GetOllamaEmbedding(string text, string model);
    }
}
