using LangChain.DocumentLoaders;
using LangChain.Extensions;
using LangChain.Schema;
using LangChain.Splitters.Text;
using Tiktoken;

namespace RAG_LLM_INTEGRATION.Helpers
{
    public class TokenTextSplitter
    {
        private readonly int _chunkSize;
        private readonly int _chunkOverlap;
        private readonly Encoder _encoder;

        public TokenTextSplitter(int chunkSize, int chunkOverlap)
        {
            _chunkSize = chunkSize;
            _chunkOverlap = chunkOverlap;
            _encoder = ModelToEncoder.For("gpt-4"); // Use the ModelToEncoder class
        }

        public List<Document> SplitDocuments(List<Document> documents)
        {
            var chunks = new List<Document>();

            foreach (var document in documents)
            {
                var tokens = _encoder.Encode(document.PageContent);

                for (int i = 0; i < tokens.Count; i += _chunkSize - _chunkOverlap)
                {
                    var end = Math.Min(i + _chunkSize, tokens.Count);
                    var chunkTokens = tokens.Skip(i).Take(end - i).ToList();
                    var chunkText = _encoder.Decode(chunkTokens.ToArray());

                    chunks.Add(new Document(chunkText, new Dictionary<string, object>()));
                }
            }

            return chunks;
        }
    }
}
