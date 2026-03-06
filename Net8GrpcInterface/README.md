# Net8GrpcInterface

.NET 8 gRPC sample where endpoints delegate business logic through interfaces and a separate .NET client consumes them.

## Projects

- `Net8GrpcInterface/`
  - `Services/IGreeterContract.cs` and `Services/GreeterContract.cs` for the hello contract.
  - `Services/IFarewellContract.cs` and `Services/FarewellContract.cs` for the goodbye contract.
  - `Services/GreeterGrpcService.cs` exposes both RPC methods through DI-backed interfaces.
- `Net8GrpcClient/`
  - Console client that calls both `SayHello` and `SayGoodbye` RPCs.

## Run server

```bash
dotnet run --project Net8GrpcInterface/Net8GrpcInterface.csproj
```

## Run client

In another terminal:

```bash
dotnet run --project Net8GrpcClient/Net8GrpcClient.csproj
```

Optional endpoint argument:

```bash
dotnet run --project Net8GrpcClient/Net8GrpcClient.csproj -- https://localhost:5001
```
