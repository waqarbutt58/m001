using Grpc.Core;

namespace Net8GrpcInterface.Services;

public sealed class GreeterGrpcService(IGreeterContract greeterContract) : Greeter.GreeterBase
{
    public override Task<HelloReply> SayHello(HelloRequest request, ServerCallContext context)
    {
        return Task.FromResult(new HelloReply
        {
            Message = greeterContract.BuildGreeting(request.Name)
        });
    }
}
