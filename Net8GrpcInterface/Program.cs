using Net8GrpcInterface.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddGrpc();
builder.Services.AddScoped<IGreeterContract, GreeterContract>();
builder.Services.AddScoped<IFarewellContract, FarewellContract>();

var app = builder.Build();

app.MapGrpcService<GreeterGrpcService>();
app.MapGet("/", () => "Use a gRPC client to communicate with this endpoint.");

app.Run();
