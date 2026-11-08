"""Stitching of single-pattern reconstructions into the full test field."""

import numpy as np
from skimage.transform import resize


class AdaptiveImageStitcher:
    """Blend overlapping scan patches using center-weighted averaging.

    Each patch is weighted by (1 - d / d_max)^2, d being the distance from the
    patch center; overlap // 2 pixels are cropped at the border.
    """

    def __init__(self, point_size, overlap):
        self.point_size = point_size
        self.overlap = overlap

    def stitch(self, data, grid_size):
        tile_size = data.shape[1]
        composite_size = grid_size * self.point_size + self.overlap
        composite = np.zeros((composite_size, composite_size), dtype=float)
        weight_map = np.zeros_like(composite)
        tiles = np.asarray(data, dtype=float).reshape(
            grid_size, grid_size, tile_size, tile_size
        )

        for row in range(grid_size):
            for column in range(grid_size):
                x_start = self.point_size * row
                y_start = self.point_size * column
                x_end = x_start + tile_size
                y_end = y_start + tile_size
                tile = tiles[row, column]
                weight = self._adaptive_weight(tile.shape)
                composite[x_start:x_end, y_start:y_end] += tile * weight
                weight_map[x_start:x_end, y_start:y_end] += weight

        normalized = np.divide(
            composite,
            weight_map,
            out=np.zeros_like(composite),
            where=weight_map != 0,
        )
        crop = self.overlap // 2
        return normalized[crop:-crop, crop:-crop]

    @staticmethod
    def _adaptive_weight(shape):
        y, x = np.ogrid[: shape[0], : shape[1]]
        center = np.array(shape) / 2
        distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        return (1 - distance / np.max(distance)) ** 2


class ExperimentalImageStitcher:
    """Apply the scan geometry of the experimental data.

    Each 64 x 64 prediction contributes its central 18 x 18 region at a scan
    step of three pixels. Overlaps are averaged, borders are cropped, and the
    result is resized to 60 x 60.
    """

    def __init__(self, point_size=3, patch_size=64, output_size=(60, 60)):
        self.point_size = point_size
        self.patch_size = patch_size
        self.output_size = output_size

    def stitch(self, data, grid_size):
        overlap = 6 * self.point_size
        patch_center = self.patch_size // 2
        composite_size = grid_size * self.point_size + overlap
        composite = np.zeros((composite_size, composite_size), dtype=float)
        count = np.zeros_like(composite)
        patches = np.asarray(data, dtype=float).reshape(
            grid_size, grid_size, self.patch_size, self.patch_size
        )
        start = patch_center - overlap // 2
        end = patch_center + overlap // 2
        patches = patches[:, :, start:end, start:end]

        for row in range(grid_size):
            for column in range(grid_size):
                x_start = self.point_size * row
                y_start = self.point_size * column
                x_end = x_start + overlap
                y_end = y_start + overlap
                composite[x_start:x_end, y_start:y_end] += patches[row, column]
                count[x_start:x_end, y_start:y_end] += 1

        half_overlap = overlap // 2
        composite = composite[half_overlap:-half_overlap, half_overlap:-half_overlap]
        count = count[half_overlap:-half_overlap, half_overlap:-half_overlap]
        stitched = composite / count
        return resize(
            stitched,
            self.output_size,
            preserve_range=True,
            anti_aliasing=True,
        )


def stitcher_for(dataset):
    """AD_LTEM: scan step 8 px, overlap 40 px. Experimental: step 3 px, 64 x 64 patches, 60 x 60 output."""

    return AdaptiveImageStitcher(point_size=8, overlap=40) if dataset == "ad_ltem" else ExperimentalImageStitcher()
