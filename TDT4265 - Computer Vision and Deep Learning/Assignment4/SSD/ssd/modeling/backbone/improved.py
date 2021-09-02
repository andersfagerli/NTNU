import torch


class ImprovedModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 7 feature maps, with the sizes:
    [shape(-1, output_channels[0], 76, 76),
     shape(-1, output_channels[1], 38, 38)
     shape(-1, output_channels[2], 19, 19),
     shape(-1, output_channels[3], 10, 10),
     shape(-1, output_channels[4], 5, 5),
     shape(-1, output_channels[5], 3, 3),
     shape(-1, output_channels[6], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        num_filters = 32
        self.feature_map_zero = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.BatchNorm2d(
                num_features=num_filters
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(num_filters, num_filters*2, 3, 1, 1),
            torch.nn.BatchNorm2d(num_filters*2),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(num_filters*2, num_filters*2, 3, 1, 1),
            torch.nn.BatchNorm2d(num_filters*2),
            torch.nn.Conv2d(num_filters*2, self.output_channels[0], 2, 2, 1)
        )

        self.feature_map_one = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[0], self.output_channels[0], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[0]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[0], self.output_channels[0], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[0]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[0], self.output_channels[1], 4, 2, 1),
            torch.nn.BatchNorm2d(self.output_channels[1]),
        )

        self.feature_map_two = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[1], self.output_channels[1], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[1], self.output_channels[1], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[1], self.output_channels[2], 3, 2, 1),
            torch.nn.BatchNorm2d(self.output_channels[2]),
        )

        self.feature_map_three = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[2], self.output_channels[2], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[2]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[2], self.output_channels[2], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[2]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[2], self.output_channels[3], 3, 2, 1),
            torch.nn.BatchNorm2d(self.output_channels[3]),
        )

        self.feature_map_four = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[3], self.output_channels[3], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[3]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[3], self.output_channels[3], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[3]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[3], self.output_channels[4], 3, 2, 1),
            torch.nn.BatchNorm2d(self.output_channels[4]),
        )

        self.feature_map_five = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[4], self.output_channels[4], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[4]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[4], self.output_channels[4], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[4]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[4], self.output_channels[5], 3, 2, 1),
            torch.nn.BatchNorm2d(self.output_channels[5]),
        )
        
        self.feature_map_six = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[5], self.output_channels[5], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[5]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[5], self.output_channels[5], 3, 1, 1),
            torch.nn.BatchNorm2d(self.output_channels[5]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_channels[5], self.output_channels[6], 3, 1, 0)
        )

        self.feature_maps = [
            self.feature_map_zero,
            self.feature_map_one,
            self.feature_map_two,
            self.feature_map_three,
            self.feature_map_four,
            self.feature_map_five,
            self.feature_map_six
        ]

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        out_features = []
        out_features.append(self.feature_map_zero(x))
        for i in range(1, len(self.feature_maps)):
            out_features.append(self.feature_maps[i](out_features[i-1]))

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)