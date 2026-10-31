import torch

class GaussHeatmapGenerator:
    def __init__(self, img_size=(640, 640), sigma_ratio=1.2):
        self.img_size = img_size
        self.sigma_ratio = sigma_ratio

    def __call__(self, bboxes, weights=None, normalize=True):

        H, W = self.img_size
        device = bboxes.device
        dtype = bboxes.dtype

        heatmap = torch.zeros((H, W), dtype=dtype, device=device)

        if bboxes.numel() == 0:
            return heatmap.unsqueeze(0)

        if weights is None:
            weights = torch.ones((bboxes.shape[0],), dtype=dtype, device=device)
        else:
            weights = weights.to(device=device, dtype=dtype)

        for i, box in enumerate(bboxes):
            x_center, y_center, width, height = box
            amp = weights[i]

            x_center_px = int((x_center * W).clamp(0, W - 1).item())
            y_center_px = int((y_center * H).clamp(0, H - 1).item())
            w_px = max(int((width * W).item()), 1)
            h_px = max(int((height * H).item()), 1)

            sigma_x = max(w_px * self.sigma_ratio, 1.0)
            sigma_y = max(h_px * self.sigma_ratio, 1.0)

            kernel = self._gaussian_kernel(sigma_x, sigma_y, device=device, dtype=dtype)
            if kernel.numel() == 0:
                continue

            k_h, k_w = kernel.shape
            radius_x = k_w // 2
            radius_y = k_h // 2

            x_start = max(x_center_px - radius_x, 0)
            y_start = max(y_center_px - radius_y, 0)
            x_end = min(x_center_px + radius_x + 1, W)
            y_end = min(y_center_px + radius_y + 1, H)

            k_start_x = max(radius_x - (x_center_px - x_start), 0)
            k_start_y = max(radius_y - (y_center_px - y_start), 0)
            k_end_x = k_w - max((x_center_px + radius_x + 1) - x_end, 0)
            k_end_y = k_h - max((y_center_px + radius_y + 1) - y_end, 0)

            kernel_cropped = kernel[k_start_y:k_end_y, k_start_x:k_end_x]
            if kernel_cropped.numel() == 0:
                continue

            patch_h = y_end - y_start
            patch_w = x_end - x_start
            if kernel_cropped.shape != (patch_h, patch_w):
                kernel_cropped = kernel_cropped[:patch_h, :patch_w]

            heatmap[y_start:y_end, x_start:x_end] += amp * kernel_cropped

        if normalize and heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap.unsqueeze(0)

    def _gaussian_kernel(self, sigma_x, sigma_y, device, dtype):
        sigma_x = max(float(sigma_x), 0.1)
        sigma_y = max(float(sigma_y), 0.1)
        kernel_w = int(6 * sigma_x) + 1
        kernel_h = int(6 * sigma_y) + 1
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1

        x = torch.arange(kernel_w, dtype=dtype, device=device) - (kernel_w // 2)
        y = torch.arange(kernel_h, dtype=dtype, device=device) - (kernel_h // 2)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        kernel = torch.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))

        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        return kernel
