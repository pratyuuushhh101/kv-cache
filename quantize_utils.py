import torch

def quantize_tensor(x: torch.Tensor, dim: int = None):
    if dim is None:
        x_min = x.min()
        x_max = x.max()
    else:
        dim = dim if dim >= 0 else x.dim() + dim  # normalize negative dim
        reduce_dims = [d for d in range(x.dim()) if d != dim]
        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)

    scale = (x_max - x_min) / 255.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero_point = x_min

    x_quantized = ((x - zero_point) / scale).round().clamp(0, 255).to(torch.uint8)

    return x_quantized, scale, zero_point

def dequantize_tensor(x_quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    return x_quantized.to(torch.float32) * scale + zero_point

def quantize_tensor_symmetric(x: torch.Tensor, dim: int = None):
    """
    Symmetric quantization: zero_point is fixed at 0, scale derived from
    max absolute value. Maps to signed int8 range [-127, 127].
    Better suited when data is roughly centered around zero (common for
    attention keys/values after layernorm).
    """
    if dim is None:
        max_abs = x.abs().max()
    else:
        dim = dim if dim >= 0 else x.dim() + dim
        reduce_dims = [d for d in range(x.dim()) if d != dim]
        max_abs = x.abs().amax(dim=reduce_dims, keepdim=True)

    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    x_quantized = (x / scale).round().clamp(-127, 127).to(torch.int8)

    return x_quantized, scale


def dequantize_tensor_symmetric(x_quantized: torch.Tensor, scale: torch.Tensor):
    return x_quantized.to(torch.float32) * scale

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 12, 10, 64)

    print("=== Asymmetric, whole-tensor ===")
    q, s, z = quantize_tensor(x, dim=None)
    r = dequantize_tensor(q, s, z)
    print(f"Mean abs error: {(x - r).abs().mean().item():.6f}")

    print("\n=== Asymmetric, per-channel (dim=-1) ===")
    q, s, z = quantize_tensor(x, dim=-1)
    r = dequantize_tensor(q, s, z)
    print(f"Mean abs error: {(x - r).abs().mean().item():.6f}")

    print("\n=== Asymmetric, per-token (dim=2) ===")
    q, s, z = quantize_tensor(x, dim=2)
    r = dequantize_tensor(q, s, z)
    print(f"Mean abs error: {(x - r).abs().mean().item():.6f}")

    print("\n=== Symmetric, whole-tensor ===")
    q, s = quantize_tensor_symmetric(x, dim=None)
    r = dequantize_tensor_symmetric(q, s)
    print(f"Mean abs error: {(x - r).abs().mean().item():.6f}")

    print("\n=== Symmetric, per-channel (dim=-1) ===")
    q, s = quantize_tensor_symmetric(x, dim=-1)
    r = dequantize_tensor_symmetric(q, s)
    print(f"Mean abs error: {(x - r).abs().mean().item():.6f}")