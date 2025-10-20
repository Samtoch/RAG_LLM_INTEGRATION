using LangChain.DocumentLoaders;
using LangChain.Schema;
using UglyToad.PdfPig;

namespace RAG_LLM_INTEGRATION.Helpers
{
    public class PdfDocumentLoader
    {
        private readonly string _filePath;

        public PdfDocumentLoader(string filePath)
        {
            _filePath = filePath;
        }

        public Task<List<Document>> LoadAsync()
        {
            var documents = new List<Document>();

            using (var pdf = PdfDocument.Open(_filePath))
            {
                foreach (var page in pdf.GetPages())
                {
                    var text = page.Text;
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        documents.Add(new Document(text));
                    }
                }
            }

            return Task.FromResult(documents);
        }
    }

}
