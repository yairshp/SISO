import torch
import open_clip
import kornia.geometry.transform as KTF
import kornia.augmentation as KTA

# import kornia.enhance as KE


class AddMarginProduct(torch.nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = torch.nn.functional.linear(
            torch.nn.functional.normalize(input),
            torch.nn.functional.normalize(self.weight),
        )

        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device="cuda")

        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
            (1.0 - one_hot) * cosine
        )  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, s={self.s}, m={self.m})"
        )


class add_layer_model(torch.nn.Module):
    def __init__(self, backbone):
        super(add_layer_model, self).__init__()
        self.backbone = backbone
        self.fc1 = torch.nn.Linear(1024, 64)
        # self.bn = torch.nn.BatchNorm1d(64)
        self.fc2 = AddMarginProduct(
            64, 10000, s=30, m=0.55
        )  # torch.nn.Linear(64, classes_num, bias = False) #AddMarginProduct(64, classes_num, s=30, m=0.35)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        fc1_out = self.fc1(x)
        return fc1_out


def get_ir_model_and_transforms(model_path, device) -> dict:
    """
    Returns the IR models.

    Args:
        config (dict): The configuration for the IR models.

    Returns:
        dict: The IR models.
    """
    backbone, _, preprocess_val = open_clip.create_model_and_transforms("ViT-H-14-290")
    backbone = backbone.visual
    model = add_layer_model(backbone)
    weight_try = torch.load(
        model_path,
        map_location=device,
    )
    weight_clear = {}
    for i in weight_try.items():
        weight_clear[i[0].split("module.")[-1]] = i[1]
    model.load_state_dict(weight_clear, strict=True)

    differentiable_transforms = get_differentiable_transforms(preprocess_val.transforms)
    return model, differentiable_transforms


def get_differentiable_transforms(curr_transforms):
    resize_size = curr_transforms[0].size
    center_crop_size = curr_transforms[1].size
    normalize_mean = curr_transforms[-1].mean
    normalize_std = curr_transforms[-1].std

    return torch.nn.Sequential(
        KTF.Resize(resize_size, interpolation="bicubic"),
        KTA.CenterCrop(center_crop_size),
        KTA.Normalize(
            mean=torch.tensor(normalize_mean),
            std=torch.tensor(normalize_std),
        ),
    )


def get_ir_features(model, transforms, image):
    model.eval()
    image = transforms(image)
    image = image.to("cuda")
    features = model(image)
    return features


def get_ir_features_sample_cos_sim(
    model, transforms, query_image_inputs, key_image_inputs
):
    query_image_features = get_ir_features(model, transforms, query_image_inputs)
    key_image_features = get_ir_features(model, transforms, key_image_inputs)
    cos_sim = torch.nn.functional.cosine_similarity(
        query_image_features, key_image_features, dim=1
    )
    return cos_sim


def get_ir_features_negative_mean_cos_sim(
    model, transforms, query_image, key_images_features
):
    cos_sims = []
    # if key_images_features is not a list, convert it to a list
    if not isinstance(key_images_features, list):
        key_images_features = [key_images_features]
    key_images_features_clones = [
        key_image_fearures.detach().clone()
        for key_image_fearures in key_images_features
    ]
    query_image_features = get_ir_features(model, transforms, query_image)
    for key_image_features in key_images_features_clones:
        cos_sim = torch.nn.functional.cosine_similarity(
            query_image_features.squeeze(), key_image_features.squeeze(), dim=0
        )
        cos_sims.append(cos_sim)
    mean_cos_sim = torch.mean(torch.stack(cos_sims))
    return -mean_cos_sim
