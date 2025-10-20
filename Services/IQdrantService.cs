namespace RAG_LLM_INTEGRATION.Services
{
    public interface IQdrantService
    {
        Task<(bool Exists, string Message)> CollectionExists(string collectionName);
        Task<(bool Success, string Message)> CreateCollection(string collectionName);
        Task<(bool Success, string Message)> UploadVector(string collectionName, int id, float[] vector, string name);
        Task<float[]> GetOllamaEmbedding(string text, string model);
    }
}
