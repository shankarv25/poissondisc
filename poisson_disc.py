from __future__ import division
import numpy as np
import itertools

class PoissonDiscSampler:
    def __init__(self, r, limits):
        self.r = r
        self.limits = np.array(limits)
        self.n = len(limits)
        self.cell_size = r/np.sqrt(self.n)
        dims = []
        for i in range(self.n):
            dim = np.ceil((self.limits[i][1] - self.limits[i][0])/self.cell_size)
            dims.append(int(dim))
        self.grid = -1*np.ones(tuple(dims), dtype=np.int)
        self.ranges = self.limits[:,1] - self.limits[:,0]
        self.offsets = self.limits[:, 0]
        self.samples = []
        self.active_list = []
        
    def sample(self, num_samples, k):
        self.samples = []
        self.active_list = []

        x_0 = self.ranges*np.random.rand(self.n) + self.offsets
        index = tuple(self.get_cell(x_0))

        self.grid[index] = 0
        self.samples.append(x_0)
        self.active_list.append(index)

        while len(self.samples) < num_samples and len(self.active_list) > 0:
            idx = np.random.choice(range(len(self.active_list)))
            cell = self.active_list[idx]
            center = self.samples[self.grid[tuple(cell)]]
            points = self.sample_points(center, k)
            n_accepted = self.verify_samples(points)
            if n_accepted == 0:
                self.active_list.remove(cell)
        return self.samples
    
    def sample_points(self, center, k=100):
        x = np.random.normal(size=(k, self.n))
        x = x/np.sqrt((x**2).sum(axis=1))[:, np.newaxis]
        r_samples = np.random.uniform(self.r, 2*self.r, size=k)
        points = r_samples[:, np.newaxis]*x
        return center + points
    
    def get_cell(self, sample):
        return np.floor_divide(sample, self.cell_size).astype(np.int)
    
    def get_neighbours(self, cell):
        indices = zip(cell-1, cell, cell + 1)
        neighbours = list(itertools.product(*indices))
        neighbours.remove(tuple(cell))
        valid_idx = ((neighbours >= self.offsets) & (neighbours < np.array(self.grid.shape))).all(axis=1).nonzero()[0]
        neighbours = [neighbours[i] for i in valid_idx]
        return neighbours
    
    def verify_samples(self, points):
        cells = self.get_cell(points)
        num_accepted = 0
        for i in range(len(points)):
            point = points[i]
            cell = cells[i]

            if (point < self.offsets).any() or (point > self.limits[:, 1]).any():
                continue

            if self.grid[tuple(cell)] >= 0:
                continue

            nearby_points = []

            neighbours = self.get_neighbours(cell)

            for neighbour in neighbours:
                sample_idx = self.grid[neighbour]
                if sample_idx >= 0:
                    nearby_point = self.samples[sample_idx]
                    nearby_points.append(nearby_point)

            nearby_points = np.array(nearby_points)
            if len(nearby_points) == 0 or (((nearby_points - point)**2).sum(axis=1) >= self.r**2).all():
                self.grid[tuple(cell)] = len(self.samples)
                self.samples.append(point)
                self.active_list.append(tuple(cell))
                num_accepted += 1

        return num_accepted
