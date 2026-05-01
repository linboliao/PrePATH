import torch


class TorchVahadaneNormalizer:
    def __init__(self, device='cpu', stain_threshold=0.15):
        self.device = device
        self.stain_threshold = stain_threshold
        self.stain_matrix_target = None
        self.target_concentrations_max = None

    def _optical_density(self, tensor):
        """将 RGB 图像转换为光学密度 (OD) 空间"""
        return -torch.log(tensor + 1e-6)

    def _get_stain_matrix(self, od_tensor):
        """
        基于简单的非负约束估计染色矩阵
        在纯 Tensor 环境下，我们通过 SVD 或迭代投影近似 NMF 的效果
        """
        od_flat = od_tensor.reshape(-1, 3)
        mask = (od_flat > self.stain_threshold).all(dim=1)
        od_filtered = od_flat[mask]

        if od_filtered.shape[0] == 0:
            return torch.eye(3, 2).to(self.device)

        _, _, V = torch.pca_lowrank(od_filtered, q=2)

        stain_matrix = V.abs()

        stain_matrix = stain_matrix / torch.norm(stain_matrix, dim=0)
        return stain_matrix

    def fit(self, target_tensor):
        target = target_tensor.to(self.device)
        od = -torch.log(target + 1e-6).reshape(3, -1)
        self.stain_matrix_target = self._get_stain_matrix(od)
        self.target_concentrations_max = torch.quantile(
            torch.linalg.lstsq(self.stain_matrix_target, od).solution, 0.99, dim=1, keepdim=True
        )

    def normalize(self, source_tensor):
        """
        归一化源图像
        source_tensor: [3, H, W], 范围 [0, 1]
        """
        source_tensor = source_tensor.to(self.device)
        C, H, W = source_tensor.shape

        od_source = self._optical_density(source_tensor)
        od_flat = od_source.reshape(3, -1)

        stain_matrix_source = self._get_stain_matrix(od_source)

        pinv = torch.linalg.pinv(stain_matrix_source)
        concentrations_source = pinv @ od_flat

        source_concentrations_max = torch.quantile(concentrations_source, 0.99, dim=1, keepdim=True)
        concentrations_source *= (self.target_concentrations_max.unsqueeze(1) / (source_concentrations_max + 1e-6))

        od_norm = self.stain_matrix_target @ concentrations_source
        res = torch.exp(-od_norm)

        return res.reshape(3, H, W).clamp(0, 1)
