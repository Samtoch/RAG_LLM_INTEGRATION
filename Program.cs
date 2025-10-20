
using RAG_LLM_INTEGRATION.Services;

namespace RAG_LLM_INTEGRATION
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.

            builder.Services.AddControllers();
            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            builder.Services.AddScoped<ISearchService, SearchService>();
            builder.Services.AddScoped<IQdrantService, QdrantService>();
            builder.Services.AddScoped<ILangChainService, LangChainService>();
            //builder.Services.AddScoped<ILangChainService, LangChainService>();


            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseAuthorization();


            app.MapControllers();

            app.Run();
        }
    }
}
