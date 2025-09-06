import time
import importlib
import torch

from .phi_rho import PhiRhoCodebookConfig, PhiRhoCodebookQuantizedTensor


def bench(fn, iters=3):
	fn()
	if torch.backends.mps.is_available():
		torch.mps.synchronize()
	t0 = time.time()
	for _ in range(iters):
		fn()
		if torch.backends.mps.is_available():
			torch.mps.synchronize()
	return (time.time() - t0) / iters


def try_import_baseline():
	try:
		mod = importlib.import_module('torchao.quantization.prototype.codebook.codebook')
		return mod
	except Exception:
		return None


@torch.no_grad()
def local_kmeans_1d(values: torch.Tensor, k: int, iters: int = 8):
	# Simple linear init
	min_val = values.min()
	max_val = values.max()
	centers = torch.linspace(min_val, max_val, steps=k, device=values.device, dtype=values.dtype)
	for _ in range(iters):
		# Assign via bucketize midpoints for 1D
		edges = (centers[1:] + centers[:-1]) * 0.5
		assign = torch.bucketize(values, edges)
		# Update means
		sums = torch.zeros(k, device=values.device, dtype=values.dtype)
		counts = torch.zeros(k, device=values.device, dtype=torch.long)
		sums += torch.bincount(assign, weights=values, minlength=k)
		counts += torch.bincount(assign, minlength=k)
		centers = sums / torch.clamp_min(counts.to(values.dtype), 1)
		# Fill empty clusters with percentiles
		empty = counts == 0
		if empty.any():
			q = torch.linspace(0, 1, empty.sum().item() + 2, device=values.device, dtype=values.dtype)[1:-1]
			centers[empty] = torch.quantile(values, q)
	return centers


def main():
	device = 'mps' if torch.backends.mps.is_available() else 'cpu'
	N = 1_000_000
	x = torch.randn(N, device=device)
	K = 256
	cfg = PhiRhoCodebookConfig(num_codebook_entries=K)

	def run_phi_rho():
		PhiRhoCodebookQuantizedTensor.from_float(x, cfg)

	phi_s = bench(run_phi_rho)
	print({'phi_rho_from_float_s': phi_s})
	# MSE vs original
	q_phi = PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
	phi_mse = torch.mean((q_phi.to_float() - x) ** 2).item()
	print({'phi_rho_mse': phi_mse})

	baseline = try_import_baseline()
	if baseline is not None and hasattr(baseline, 'CodebookWeightOnlyConfig'):
		# Emulate their API if available
		def run_baseline():
			cfg2 = baseline.CodebookWeightOnlyConfig(num_codebook_entries=K)
			# Minimal: if their API expects quantize_ flow, we skip and call their from_float if available
			try:
				q = baseline.CodebookQuantizedTensor.from_float(x, cfg2)
			except Exception:
				pass
		base_s = bench(run_baseline)
		print({'baseline_from_float_s': base_s})
	else:
		# Fallback to local k-means baseline
		def run_local():
			centers = local_kmeans_1d(x, K, iters=8)
			assign = torch.bucketize(x, (centers[1:] + centers[:-1]) * 0.5)
			_ = centers[assign]
		base_s = bench(run_local)
		print({'baseline_from_float_s': base_s, 'baseline': 'local_kmeans'})
		# MSE vs original
		centers = local_kmeans_1d(x, K, iters=8)
		assign = torch.bucketize(x, (centers[1:] + centers[:-1]) * 0.5)
		base_mse = torch.mean((centers[assign] - x) ** 2).item()
		print({'baseline_mse': base_mse})


if __name__ == '__main__':
	main()


