namespace Net8GrpcInterface.Services;

public sealed class FarewellContract : IFarewellContract
{
    public string BuildFarewell(string name)
    {
        var safeName = string.IsNullOrWhiteSpace(name) ? "friend" : name.Trim();
        return $"Goodbye {safeName}";
    }
}
