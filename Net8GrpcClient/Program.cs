using Grpc.Net.Client;
using Net8GrpcInterface;

var address = args.Length > 0 ? args[0] : "https://localhost:5001";

using var channel = GrpcChannel.ForAddress(address);
var client = new Greeter.GreeterClient(channel);

var helloResponse = await client.SayHelloAsync(new HelloRequest
{
    Name = "gRPC client"
});

var goodbyeResponse = await client.SayGoodbyeAsync(new GoodbyeRequest
{
    Name = "gRPC client"
});

Console.WriteLine($"Server says hello: {helloResponse.Message}");
Console.WriteLine($"Server says goodbye: {goodbyeResponse.Message}");
