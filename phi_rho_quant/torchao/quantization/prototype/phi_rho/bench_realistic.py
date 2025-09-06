import os
import time
import importlib
import torch

from . import PhiRhoCodebookConfig, PhiRhoCodebookQuantizedTensor


def bench(fn, iters=3, device: str | None = None):
	fn()
	if device == 'mps':
		torch.mps.synchronize()
	t0 = time.time()
	for _ in range(iters):
		fn()
		if device == 'mps':
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
	min_val = values.min()
	max_val = values.max()
	centers = torch.linspace(min_val, max_val, steps=k, device=values.device, dtype=values.dtype)
	for _ in range(iters):
		edges = (centers[1:] + centers[:-1]) * 0.5
		assign = torch.bucketize(values, edges)
		sums = torch.bincount(assign, weights=values, minlength=k)
		counts = torch.bincount(assign, minlength=k)
		centers = sums / torch.clamp_min(counts.to(values.dtype), 1)
		empty = counts == 0
		if empty.any():
			# Lightweight fallback: linearly space within observed range
			mn = values.min()
			mx = values.max()
			fill = torch.linspace(mn, mx, steps=empty.sum().item() + 2, device=values.device, dtype=values.dtype)[1:-1]
			centers[empty] = fill
	return centers


def run_suite(device, include_baseline: bool = True):
	# Common realistic shapes (weights/activations)
	shapes = {
		'linear_w_4k_4k': (4096, 4096),
		'linear_w_8k_4k': (8192, 4096),
		'act_batch_8_4k': (8, 4096),
		'act_batch_32_4k': (32, 4096),
	}
	Ks = [64, 128, 256]
	results = []
	baseline = try_import_baseline() if include_baseline else None
	for name, shape in shapes.items():
		x = torch.randn(shape, device=device)
		for K in Ks:
			cfg = PhiRhoCodebookConfig(num_codebook_entries=K)
			cfg_hybrid = PhiRhoCodebookConfig(num_codebook_entries=K, lloyd_refine_steps=1)
			def run():
				PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
			ts = bench(run, device=device)
			q = PhiRhoCodebookQuantizedTensor.from_float(x, cfg)
			phi_mse = torch.mean((q.to_float() - x) ** 2).item()
			# Hybrid refinement
			def run_h():
				PhiRhoCodebookQuantizedTensor.from_float(x, cfg_hybrid)
			ts_h = bench(run_h, device=device)
			qh = PhiRhoCodebookQuantizedTensor.from_float(x, cfg_hybrid)
			phi_h_mse = torch.mean((qh.to_float() - x) ** 2).item()
			# Baseline timing and mse
			if include_baseline and baseline is not None and hasattr(baseline, 'CodebookWeightOnlyConfig'):
				def run_base():
					cfg2 = baseline.CodebookWeightOnlyConfig(num_codebook_entries=K)
					try:
						_ = baseline.CodebookQuantizedTensor.from_float(x, cfg2)
					except Exception:
						pass
				base_s = bench(run_base, device=device)
				try:
					q2 = baseline.CodebookQuantizedTensor.from_float(x, cfg2)
					base_mse = torch.mean((q2.to_float() - x) ** 2).item()
				except Exception:
					base_mse = float('nan')
				base_tag = 'torchao_codebook'
			elif include_baseline:
				def run_local():
					# Run local baseline on CPU to avoid MPS command buffer errors
					xc = x.view(-1).to('cpu')
					centers = local_kmeans_1d(xc, K, iters=6)
					assign = torch.bucketize(xc, (centers[1:] + centers[:-1]) * 0.5)
					_ = centers[assign]
				base_s = bench(run_local, device='cpu')
				xc = x.view(-1).to('cpu')
				centers = local_kmeans_1d(xc, K, iters=6)
				assign = torch.bucketize(xc, (centers[1:] + centers[:-1]) * 0.5)
				base_mse = torch.mean((centers[assign].view_as(xc) - xc) ** 2).item()
				base_tag = 'local_kmeans'
			entry = {
				'shape': name,
				'numel': x.numel(),
				'K': K,
				'phi_sec': ts,
				'phi_mse': phi_mse,
				'phi_h_sec': ts_h,
				'phi_h_mse': phi_h_mse,
			}
			if include_baseline:
				entry.update({'base_sec': base_s, 'base_mse': base_mse, 'baseline': base_tag})
			results.append(entry)
	return results


def run_model_suite(device):
	# Simulate multiple layers: repeat a few realistic weight shapes
	layer_shapes = [(4096, 4096), (8192, 4096), (4096, 4096), (8192, 4096)] * 3  # 12 layers
	K = 256
	cfg = PhiRhoCodebookConfig(num_codebook_entries=K)
	phi_total = 0.0
	base_total = 0.0
	baseline = try_import_baseline()
	for shape in layer_shapes:
		x = torch.randn(shape, device=device)
		phi_total += bench(lambda: PhiRhoCodebookQuantizedTensor.from_float(x, cfg), iters=1, device=device)
		if baseline is not None and hasattr(baseline, 'CodebookWeightOnlyConfig'):
			def run_base():
				cfg2 = baseline.CodebookWeightOnlyConfig(num_codebook_entries=K)
				try:
					_ = baseline.CodebookQuantizedTensor.from_float(x, cfg2)
				except Exception:
					pass
			base_total += bench(run_base, iters=1, device=device)
		else:
			def run_local():
				centers = local_kmeans_1d(x.view(-1), K, iters=4)
				assign = torch.bucketize(x.view(-1), (centers[1:] + centers[:-1]) * 0.5)
				_ = centers[assign]
			base_total += bench(run_local, iters=1, device='cpu')
	return {'layers': len(layer_shapes), 'K': K, 'phi_total_sec': phi_total, 'base_total_sec': base_total}


def main():
	preferred = os.getenv('RUN_DEVICE')
	device = preferred if preferred in ('cpu', 'mps') else ('mps' if torch.backends.mps.is_available() else 'cpu')
	include_baseline = os.getenv('INCLUDE_BASELINE', '1') == '1'
	res = run_suite(device, include_baseline=include_baseline)
	for r in res:
		print(r)
	print(run_model_suite(device))


if __name__ == '__main__':
	main()


