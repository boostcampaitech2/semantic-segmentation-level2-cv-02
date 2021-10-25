from albumentations.core.transforms_interface import BasicTransform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch


def elastic_transform_img(image, alpha=991, sigma=8, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(12)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_image = map_coordinates(image, indices, order=1, mode="reflect")
    return distored_image.reshape(image.shape)


def elastic_transform_mask(mask, alpha=991, sigma=8, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(12)
    mask = torch.Tensor(mask)
    mask = torch.stack([mask, mask, mask], dim=2)  # 512*512*3 이어야 같은 랜덤값으로 transform 가능
    mask = mask.numpy()
    shape = mask.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_image = map_coordinates(mask, indices, order=1, mode="reflect")

    return distored_image.reshape(mask.shape)[:, :, 0]  # 512*512*3 에서 아무거나 한장만 return


class Elastic_Transform(BasicTransform):
    def __init__(self, alpha=991, sigma=8, p=0.5, always_apply=False):
        super(Elastic_Transform, self).__init__(always_apply=always_apply, p=p)
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.always_apply = False
        self._additional_targets: Dict[str, str] = {}
        self.deterministic = False
        self.save_key = "replay"
        self.params: Dict[Any, Any] = {}
        self.replay_mode = False
        self.applied_in_replay = False

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        return elastic_transform_img(img, alpha=self.alpha, sigma=self.sigma)

    def apply_to_mask(self, mask, **params):
        return elastic_transform_mask(mask, alpha=self.alpha, sigma=self.sigma)

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}

    # def __call__(self,img):
    #    return A.geometric.transforms.ElasticTransform(p = self.p, alpha = self.alpha, sigma = self.sigma, )
