import torch

def quantize_tensor(x: torch.Tensor):
    """Quantize a float32 tensor to int8 using min-max scaling."""
    x_min = x.min()
    x_max = x.max()

    scale = (x_max - x_min) / 255.0
    zero_point = x_min

    x_quantized = ((x - zero_point) / scale).round().clamp(0, 255).to(torch.uint8)

    return x_quantized, scale, zero_point

def dequantize_tensor(x_quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """Reverse quantization back to float32."""
    return x_quantized.to(torch.float32) * scale + zero_point


if __name__ == "__main__":
    # Sanity check: quantize -> dequantize a random tensor, measure error
    torch.manual_seed(0)
    original = torch.randn(4, 8)  # small random tensor, mimics a slice of K or V

    quantized, scale, zero_point = quantize_tensor(original)
    reconstructed = dequantize_tensor(quantized, scale, zero_point)

    error = (original - reconstructed).abs()

    print("Original:\n", original)
    print("\nQuantized (uint8):\n", quantized)
    print("\nReconstructed:\n", reconstructed)
    print(f"\nMax absolute error: {error.max().item():.6f}")
    print(f"Mean absolute error: {error.mean().item():.6f}")

    # Memory comparison
    orig_bytes = original.numel() * original.element_size()
    quant_bytes = quantized.numel() * quantized.element_size()
    print(f"\nOriginal size: {orig_bytes} bytes (float32)")
    print(f"Quantized size: {quant_bytes} bytes (uint8)")
    print(f"Compression ratio: {orig_bytes / quant_bytes:.2f}x")