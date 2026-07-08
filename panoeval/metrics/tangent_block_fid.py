import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric
from torchmetrics.image.fid import NoTrainInceptionV3

def _compute_blockdiag_fid(
    mu1_list: list[Tensor],
    sigma1_list: list[Tensor],
    mu2_list: list[Tensor],
    sigma2_list: list[Tensor]
) -> Tensor:
    r"""
    Compute block-diagonal joint FID over multiple Gaussian pairs.

    Args:
        mu1_list: list of mean tensors (dim D_i) for real features per plane.
        sigma1_list: list of covariance matrices (D_i x D_i) for real features per plane.
        mu2_list: list of mean tensors (dim D_i) for fake features per plane.
        sigma2_list: list of covariance matrices (D_i x D_i) for fake features per plane.
    Returns:
        Scalar FID value.
    """
    # Concatenate means
    mu1_joint = torch.cat(mu1_list, dim=-1)
    mu2_joint = torch.cat(mu2_list, dim=-1)

    # Build block-diagonal covariances
    sigma1_joint = torch.block_diag(*sigma1_list)
    sigma2_joint = torch.block_diag(*sigma2_list)

    # Compute FID on joint Gaussian
    diff = (mu1_joint - mu2_joint).pow(2).sum()
    trace_term = (sigma1_joint.trace() + sigma2_joint.trace())

    # sqrtm term: eigenvalues of product
    # compute eigenvalues of sigma1_joint @ sigma2_joint
    evals = torch.linalg.eigvals(sigma1_joint @ sigma2_joint)
    # take real part, square root, sum
    sqrt_trace = evals.real.sqrt().sum()

    fid_val = diff + trace_term - 2 * sqrt_trace
    return fid_val


class PanoramicFrechetInceptionDistance(Metric):
    """
    FID metric for panoramic images by computing per-plane stats and using block-diagonal joint FID.
    """
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    inception: Module
    feature_network: str = "inception"

    def __init__(self, num_planes: int = 18, **kwargs):
        super().__init__(**kwargs)
        self.num_planes = num_planes
        self.plane_states = []
        # For each plane, we track sum, cov_sum, num_samples
        for i in range(num_planes):
            self.add_state(f"real_sum_{i}", torch.zeros(0), dist_reduce_fx="sum")
            self.add_state(f"real_cov_{i}", torch.zeros(0), dist_reduce_fx="sum")
            self.add_state(f"real_n_{i}", torch.tensor(0), dist_reduce_fx="sum")
            self.add_state(f"fake_sum_{i}", torch.zeros(0), dist_reduce_fx="sum")
            self.add_state(f"fake_cov_{i}", torch.zeros(0), dist_reduce_fx="sum")
            self.add_state(f"fake_n_{i}", torch.tensor(0), dist_reduce_fx="sum")

        self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(2048)])

    def update(self, features: Tensor, real: bool, plane_idx: int) -> None:
        """
        Update stats for a given plane.

        Args:
            features: Tensor of shape (N, D)
            real: bool, whether real or fake
            plane_idx: index of tangent plane [0, num_planes)
        """
        sum_state = f"real_sum_{plane_idx}" if real else f"fake_sum_{plane_idx}"
        cov_state = f"real_cov_{plane_idx}" if real else f"fake_cov_{plane_idx}"
        n_state = f"real_n_{plane_idx}" if real else f"fake_n_{plane_idx}"

        s = getattr(self, sum_state)
        cov = getattr(self, cov_state)
        n = getattr(self, n_state)

        batch_sum = features.sum(dim=0).double()
        batch_cov = features.t().double() @ features.double()
        batch_n = features.shape[0]

        if s.numel() == 0:
            # initialize sum and cov with correct dims
            setattr(self, sum_state, batch_sum)
            setattr(self, cov_state, batch_cov)
        else:
            setattr(self, sum_state, s + batch_sum)
            setattr(self, cov_state, cov + batch_cov)

        setattr(self, n_state, n + batch_n)

    def compute(self) -> Tensor:
        """
        Compute joint block-diagonal FID across all planes.
        """
        mu1_list, sigma1_list = [], []
        mu2_list, sigma2_list = [], []
        for i in range(self.num_planes):
            # real
            s = getattr(self, f"real_sum_{i}")
            cov_s = getattr(self, f"real_cov_{i}")
            n = getattr(self, f"real_n_{i}")
            mean = (s / n).unsqueeze(0)
            cov = (cov_s - n * mean.t() @ mean) / (n - 1)
            mu1_list.append(mean.squeeze(0))
            sigma1_list.append(cov)
            # fake
            s2 = getattr(self, f"fake_sum_{i}")
            cov_s2 = getattr(self, f"fake_cov_{i}")
            n2 = getattr(self, f"fake_n_{i}")
            mean2 = (s2 / n2).unsqueeze(0)
            cov2 = (cov_s2 - n2 * mean2.t() @ mean2) / (n2 - 1)
            mu2_list.append(mean2.squeeze(0))
            sigma2_list.append(cov2)

        # call blockdiag FID
        return _compute_blockdiag_fid(mu1_list, sigma1_list, mu2_list, sigma2_list)