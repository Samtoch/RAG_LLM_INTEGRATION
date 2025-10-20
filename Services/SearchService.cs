using RAG_LLM_INTEGRATION.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace RAG_LLM_INTEGRATION.Services
{
    public class SearchService : ISearchService
    {
        private readonly IQdrantService _qdrantService;
        private readonly IConfiguration _configuration;
        private string? _qdrantUrl;
        private string? _qdrantKey;
        private string? _openAIUrl;
        private string? _openAIAPIKey;
        public SearchService(IConfiguration configuration, IQdrantService qdrantService)
        {
            _qdrantService = qdrantService;
            _configuration = configuration;
            _qdrantUrl = _configuration["QdrantUrl"];
            _qdrantKey = _configuration["QdrantApiKey"];

            _openAIUrl = _configuration["OpenAIUrl"];
            _openAIAPIKey = _configuration["OpenAIApiKey"];
        }

        public async Task<(bool Success, string Message)> CreateEmbedingsForCollection(string collectionName, List<string> entries)
        {
            //string model = "all-minilm";


            // Ensure collection exists
            var (exists, msg) = await _qdrantService.CollectionExists(collectionName);
            if (!exists)
            {
                await _qdrantService.CreateCollection(collectionName);
            }

            for (int i = 0; i < entries.Count; i++)
            {
                string entry = entries[i];
                float[] embedding = await GetOpenAIEmbedding(entry);

                if (embedding != null)
                {
                    var (isPointsCreated, pointMessage) = await _qdrantService.UploadVector(collectionName, i, embedding, entry);
                    if (!isPointsCreated)
                        return (false, $"Vector point upload failed: {pointMessage}");

                    Console.WriteLine($"Uploaded: {entry}");
                }
                else
                {
                    Console.WriteLine($"Failed: {entry}");
                    return (false, $"Embedding generation failed for entry: {entry}");
                }
            }

            return (true, $"Embeddings successfully created for collection '{collectionName}'.");
        }


        public async Task<List<SearchMatch>> SemanticSearch(string collectionName, string inputText)
        {
            string model = "all-minilm";
            float[] inputVector = await GetOllamaEmbedding(inputText, model);

            if (inputVector == null)
                throw new Exception("Embedding generation failed.");

            using HttpClient client = new HttpClient();
            client.DefaultRequestHeaders.Add("api-key", _qdrantKey);

            var requestBody = new
            {
                vector = inputVector,
                top = 3,
                with_payload = true
            };

            string json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync($"{_qdrantUrl}/{collectionName}/points/search", content);
            if (!response.IsSuccessStatusCode)
            {
                throw new Exception("Search failed: " + response.StatusCode);
            }

            var responseJson = await response.Content.ReadAsStringAsync();
            using JsonDocument doc = JsonDocument.Parse(responseJson);

            var results = new List<SearchMatch>();

            foreach (var result in doc.RootElement.GetProperty("result").EnumerateArray())
            {
                var score = result.GetProperty("score").GetDouble();
                var payload = result.GetProperty("payload");
                var name = payload.GetProperty("name").GetString();

                results.Add(new SearchMatch { Name = name, Score = score });
            }

            return results;
        }

        public async Task<string> GetLlama3Answer(string prompt)
        {
            using var client = new HttpClient();

            var requestBody = new
            {
                model = "llama3",
                prompt = prompt,
                stream = false
            };

            string json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync("http://localhost:11434/api/generate", content);
            string responseJson = await response.Content.ReadAsStringAsync();
            Console.WriteLine("responseJson: " + responseJson);
            response.EnsureSuccessStatusCode();

            //string responseJson = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(responseJson);
            return doc.RootElement.GetProperty("response").GetString();
        }

        public async Task<string> GetDeepSeekAnswer(string prompt)
        {
            using var client = new HttpClient();

            // Add DeepSeek API Key to headers
            client.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("", "");

            var requestBody = new
            {
                model = "deepseek-chat", // Replace with actual DeepSeek model name if different
                messages = new[]
                {
                    new { role = "user", content = prompt }
                }
            };

            string json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync("https://api.deepseek.com/v1/chat/completions", content);
            string responseJson = await response.Content.ReadAsStringAsync();

            Console.WriteLine("responseJson: " + responseJson);
            response.EnsureSuccessStatusCode();

            using var doc = JsonDocument.Parse(responseJson);
            return doc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString();
        }

        public async Task<string> GetOpenAIGpt4oMiniAnswer(string prompt)
        {
            using var client = new HttpClient();
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _openAIAPIKey);

            var requestBody = new
            {
                model = "gpt-4o-mini",
                messages = new[]
                {
                    new { role = "user", content = prompt + "\n [Response should not be more than 450 characters]" }
                },
                temperature = 0.7
            };

            string json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync($"{_openAIUrl}/chat/completions", content);
            string responseJson = await response.Content.ReadAsStringAsync();

            Console.WriteLine("responseJson: " + responseJson);
            response.EnsureSuccessStatusCode();

            using var doc = JsonDocument.Parse(responseJson);
            return doc.RootElement
                      .GetProperty("choices")[0]
                      .GetProperty("message")
                      .GetProperty("content")
                      .GetString();
        }


        public async Task<List<SearchMatch>> SearchTopMatches(string collectionName, float[] vector, int topK)
        {
            using var client = new HttpClient();
            client.DefaultRequestHeaders.Add("api-key", _qdrantKey);

            var request = new
            {
                vector = vector,
                top = topK,
                with_payload = true
            };

            string json = JsonSerializer.Serialize(request);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync($"{_qdrantUrl}/collections/{collectionName}/points/search", content);
            if (!response.IsSuccessStatusCode)
            {
                string errorContent = await response.Content.ReadAsStringAsync();
                Console.WriteLine($"Error Response: {errorContent}");
                return null;
            }
            response.EnsureSuccessStatusCode();
            

            string responseJson = await response.Content.ReadAsStringAsync();
            using JsonDocument doc = JsonDocument.Parse(responseJson);

            var results = new List<SearchMatch>();
            foreach (var match in doc.RootElement.GetProperty("result").EnumerateArray())
            {
                string name = match.GetProperty("payload").GetProperty("name").GetString();
                double score = match.GetProperty("score").GetDouble();
                results.Add(new SearchMatch { Name = name, Score = score });
            }

            return results;
        }


        //public async Task<List<string>> SemanticSearch(string userQuery)
        //{
        //    string model = "all-minilm";

        //    userQuery = "Tell me about president Trump.";
        //    float[] userEmbedding = await GetOllamaEmbedding(userQuery, model);

        //    if (userEmbedding == null)
        //    {
        //        throw new Exception("Failed to generate embedding for user query.");
        //    }

        //    List<string> blogPostTitles = new List<string>
        //    {
        //        "AI is transforming the world.",
        //        "Machine learning enables computers to learn from data.",
        //        "Quantum computing is the future of technology.",
        //        "Deep learning is a subset of machine learning."
        //    };

        //    // Generate embeddings for each blog post title
        //    var candidateEmbeddings = await GenerateEmbeddings(blogPostTitles, model);

        //    if (candidateEmbeddings == null || candidateEmbeddings.Count == 0)
        //    {
        //        throw new Exception("Failed to generate candidate embeddings.");
        //    }

        //    // Compute cosine similarities and get the top three matches.
        //    var topMatches = candidateEmbeddings
        //        .Select(candidate => new
        //        {
        //            Text = candidate.Key,
        //            Similarity = CosineSimilarity(candidate.Value, userEmbedding)
        //        })
        //        .OrderByDescending(match => match.Similarity)
        //        .Take(3)
        //        .Select(match => match.Text)  // Only return the text
        //        //.Select(match => (match.Text, match.Similarity))  // returning Text + Similarity
        //        .ToList();

        //    return topMatches;
        //}


        // Function to compute Cosine Similarity
        public float CosineSimilarity(float[] vecA, float[] vecB)
        {
            float dotProduct = vecA.Zip(vecB, (a, b) => a * b).Sum();
            float magnitudeA = (float)Math.Sqrt(vecA.Sum(a => a * a));
            float magnitudeB = (float)Math.Sqrt(vecB.Sum(b => b * b));
            return dotProduct / (magnitudeA * magnitudeB);
        }

        // Function to generate embeddings for a single text using Ollama API
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

        public async Task<float[]> GetOpenAIEmbedding(string text)
        {
            using var client = new HttpClient();
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _openAIAPIKey);

            var requestBody = new
            {
                input = text,
                model = "text-embedding-3-small" // or text-embedding-3-large
            };

            string json = JsonSerializer.Serialize(requestBody);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync($"{_openAIUrl}/embeddings", content);

            if (response.IsSuccessStatusCode)
            {
                var responseJson = await response.Content.ReadAsStringAsync();
                using JsonDocument doc = JsonDocument.Parse(responseJson);

                var embedding = doc.RootElement
                    .GetProperty("data")[0]
                    .GetProperty("embedding");

                return JsonSerializer.Deserialize<float[]>(embedding.GetRawText());
            }

            Console.WriteLine($"Failed to get embedding: {response.StatusCode}");
            return null;
        }

    }
}

// Display results
//Console.WriteLine("\nTop matching blog post titles:");
//var topMatches_ = candidateEmbeddings
//    .Select(candidate => new
//    {
//        Text = candidate.Key,
//        Similarity = CosineSimilarity(candidate.Value, userEmbedding)
//    })
//    .OrderByDescending(match => match.Similarity)
//    .Take(3)
//    .Select(match => (match.Text, match.Similarity))  // returning Text + Similarity
//    .ToList();
//foreach (var match in topMatches_)
//{
//    Console.WriteLine($"Similarity: {match.Similarity:F4} - {match.Text}");
//}