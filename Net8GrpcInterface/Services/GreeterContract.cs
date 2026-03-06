namespace Net8GrpcInterface.Services;

public sealed class GreeterContract : IGreeterContract
{
    public string BuildGreeting(string name)
    {
        var safeName = string.IsNullOrWhiteSpace(name) ? "world" : name.Trim();
        return $"Hello {safeName}";
    }
}
