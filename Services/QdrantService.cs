
using Microsoft.AspNetCore.DataProtection.KeyManagement;
using System.Text.Json;
using System.Text;
using NLog;

namespace RAG_LLM_INTEGRATION.Services
{
    public class QdrantService : IQdrantService
    {
        private readonly IConfiguration _configuration;
        private string? _qdrantUrl;
        private string? _qdrantKey;
        private static Logger log = LogManager.GetCurrentClassLogger();

        public QdrantService(IConfiguration configuration)
        {
            _configuration = configuration;
            _qdrantUrl = _configuration["QdrantUrl"];
            _qdrantKey = _configuration["QdrantApiKey"];
        }

        public async Task<(bool Success, string Message)> CreateCollection(string collectionName)
        {
            try
            {
                using HttpClient client = new HttpClient();
                client.DefaultRequestHeaders.Add("api-key", _qdrantKey);

                var request = new
                {
                    vectors = new
                    {
                        size = 1536,   // 384 for 'all-minilm', 1536 for 'text-embedding-3-small'
                        distance = "Cosine"
                    }
                };

                string json = JsonSerializer.Serialize(request);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await client.PutAsync($"{_qdrantUrl}/collections/{collectionName}", content);

                if (!response.IsSuccessStatusCode)
                {
                    string errorContent = await response.Content.ReadAsStringAsync();
                    log.Error($"Error Response: {errorContent}");
                    return (false, $"Failed to create collection: {errorContent}");
                }

                log.Info($"Collection Created: {response.StatusCode}");
                return (true, $"Collection '{collectionName}' created successfully.");
            }
            catch (Exception ex)
            {
                log.Error($"Collection Creation Failed: {ex.Message}");
                return (false, $"Exception: {ex.Message}");
            }
        }


        public async Task<(bool Success, string Message)> UploadVector(string collectionName, int id, float[] vector, string name)
        {
            try
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

                var response = await client.PutAsync($"{_qdrantUrl}/collections/{collectionName}/points", content);
                log.Info($"Uploaded {name}: {response.IsSuccessStatusCode}");

                if (!response.IsSuccessStatusCode)
                {
                    string errorContent = await response.Content.ReadAsStringAsync();
                    log.Error($"Error Response: {errorContent}");
                    return (false, $"Failed to create points: {errorContent}");
                }
                return (true, $"Points '{collectionName}' created successfully.");
            }
            catch (Exception ex)
            {
                log.Error($"Vector Points Upload Failed: {ex.Message}");
                return (false, $"Exception: {ex.Message}");
            }
        }

        public async Task<(bool Exists, string Message)> CollectionExists(string collectionName)
        {
            try
            {
                using HttpClient client = new HttpClient();
                client.DefaultRequestHeaders.Add("api-key", _qdrantKey);

                var response = await client.GetAsync($"{_qdrantUrl}/collections/{collectionName}");

                if (response.IsSuccessStatusCode)
                {
                    return (true, $"Collection '{collectionName}' exists.");
                }
                else if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
                {
                    return (false, $"Collection '{collectionName}' does not exist.");
                }
                else
                {
                    string errorContent = await response.Content.ReadAsStringAsync();
                    return (false, $"Failed to check collection: {errorContent}");
                }
            }
            catch (Exception ex)
            {
                log.Error($"Collection Existence Check Failed: {ex.Message}");
                return (false, $"Exception while checking collection: {ex.Message}");
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

            log.Error("Failed to get embeddings.");
            return null;
        }

    }
}
