"""Bounded exact cache for fixed client UV rasterization, independent of XYZ.

The caller owns this object and can warm its 17 mouth-filter sampling grids at
service startup. Keys include every raster input. No service defaults, source
selection, camera, or photo observations are changed by this cache.
"""
from collections import OrderedDict
import hashlib

import numpy as np


class ClientUvLookupCache:
    def __init__(self, rasterizer, *, maximumEntries=17):
        if not isinstance(maximumEntries, int) or maximumEntries < 1:
            raise ValueError('maximumEntries must be a positive integer')
        self.rasterizer = rasterizer
        self.maximumEntries = maximumEntries
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def __call__(self, geometry, resolution):
        uv = np.asarray(geometry['uvs'], np.float64)
        faces = np.asarray(geometry.get('uvFaces', geometry['faces']))
        if faces.shape != np.asarray(geometry['faces']).shape:
            raise ValueError('UV triangles must match geometry face order')
        if not isinstance(resolution, (int, np.integer)) or resolution < 2:
            raise ValueError('resolution must be an integer >= 2')
        key = (int(resolution), geometry.get('uvConvention', 'topLeftPixelCenters'),
               uv.shape, faces.shape, faces.dtype.str,
               hashlib.sha256(uv.tobytes()).digest(), hashlib.sha256(faces.tobytes()).digest())
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        result = self.rasterizer(geometry, resolution)
        for value in result[0].values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        self.entries[key] = result
        while len(self.entries) > self.maximumEntries:
            self.entries.popitem(last=False)
        return result

    def warmMouthGrids(self, geometry, resolution):
        self(geometry, resolution)
        for dy in (-.75, -.25, .25, .75):
            for dx in (-.75, -.25, .25, .75):
                shifted = {**geometry, 'uvs': geometry['uvs']-np.array([dx, dy])/resolution}
                self(shifted, resolution)

    def diagnostics(self):
        return {'entries': len(self.entries), 'hits': self.hits, 'misses': self.misses,
            'arrayBytes': sum(value.nbytes for result in self.entries.values()
                              for value in result[0].values() if isinstance(value, np.ndarray)),
            'scope': 'UV coordinates, topology, convention and resolution only; no photo or geometry cache'}
