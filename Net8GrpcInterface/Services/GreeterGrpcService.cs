using Grpc.Core;

namespace Net8GrpcInterface.Services;

public sealed class GreeterGrpcService(
    IGreeterContract greeterContract,
    IFarewellContract farewellContract) : Greeter.GreeterBase
{
    public override Task<HelloReply> SayHello(HelloRequest request, ServerCallContext context)
    {
        return Task.FromResult(new HelloReply
        {
            Message = greeterContract.BuildGreeting(request.Name)
        });
    }

    public override Task<GoodbyeReply> SayGoodbye(GoodbyeRequest request, ServerCallContext context)
    {
        return Task.FromResult(new GoodbyeReply
        {
            Message = farewellContract.BuildFarewell(request.Name)
        });
    }
}
