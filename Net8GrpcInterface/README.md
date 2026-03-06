# Net8GrpcInterface

Minimal .NET 8 gRPC service where the endpoint delegates business logic through an interface.

## Structure

- `Services/IGreeterContract.cs` defines the interface.
- `Services/GreeterContract.cs` implements the interface.
- `Services/GreeterGrpcService.cs` exposes the implementation through the `Greeter` gRPC endpoint.

## Run

```bash
dotnet restore
dotnet run --project Net8GrpcInterface/Net8GrpcInterface.csproj
```

By default, ASP.NET Core listens on `https://localhost:5001` and exposes the `Greeter/SayHello` RPC.
