
using Microsoft.AspNetCore.DataProtection.KeyManagement;
using System.Text.Json;
using System.Text;

namespace RAG_LLM_INTEGRATION.Services
{
    public class QdrantService : IQdrantService
    {
        private readonly IConfiguration _configuration;
        private string? _qdrantUrl;
        private string? _qdrantKey;
        public QdrantService(IConfiguration configuration)
        {
            _configuration = configuration;
            _qdrantUrl = _configuration["QdrantUrl"];
            _qdrantKey = _configuration["QdrantApiKey"];
        }

        public async Task CreateCollection(string collectionName)
        {
            using HttpClient client = new HttpClient();
            client.DefaultRequestHeaders.Add("api-key", _qdrantKey);

            var request = new
            {
                vectors = new
                {
                    size = 384, // match 'all-minilm' output dim
                    distance = "Cosine"
                }
            };

            string json = JsonSerializer.Serialize(request);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PutAsync($"{_qdrantUrl}/{collectionName}", content);
            Console.WriteLine($"Collection Created: {response.IsSuccessStatusCode}");
        }

        public async Task UploadVector(string collectionName, int id, float[] vector, string name)
        {
            using HttpClient client = new HttpClient();
            client.Timeout = TimeSpan.FromMinutes(5);

            client.DefaultRequestHeaders.Add("api-key", _qdrantKey);

            var point = new
            {
                id = id,
                vector = vector,
                payload = new { name = name }
            };

            var requestBody = new
            {
                points = new[] { point }
            };

            string json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PutAsync($"{_qdrantUrl}/{collectionName}/points", content);
            Console.WriteLine($"Uploaded {name}: {response.IsSuccessStatusCode}");

            if (!response.IsSuccessStatusCode)
            {
                string errorContent = await response.Content.ReadAsStringAsync();
                Console.WriteLine($"Error Response: {errorContent}");
            }
        }

        public async Task<float[]> GetOllamaEmbedding(string text, string model)
        {
            using HttpClient client = new HttpClient();
            var requestBody = new
            {
                model = model,
                prompt = text,
                options = new { }
            };

            string json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            HttpResponseMessage response = await client.PostAsync("http://127.0.0.1:11434/api/embeddings", content);

            if (response.IsSuccessStatusCode)
            {
                string responseJson = await response.Content.ReadAsStringAsync();
                using JsonDocument doc = JsonDocument.Parse(responseJson);
                JsonElement root = doc.RootElement;

                if (root.TryGetProperty("embedding", out JsonElement embeddingElement))
                {
                    return JsonSerializer.Deserialize<float[]>(embeddingElement.GetRawText());
                }
            }

            Console.WriteLine("Failed to get embeddings.");
            return null;
        }

    }
}
