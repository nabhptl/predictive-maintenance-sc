"""Metaheuristics used in the project: a Genetic Algorithm and a Binary PSO (for comparison)."""
import numpy as np


class GeneticAlgorithm:
    """Generic GA. Each gene is an integer in range(gene_sizes[i]).

    gene_sizes = [2] * n_sensors       -> binary chromosome (feature selection)
    gene_sizes = [5, 4, 3]             -> index into lists of options (hyper-parameter tuning)
    fitness(genes) must return a number to MINIMISE.
    """

    def __init__(self, gene_sizes, fitness, pop_size=20, generations=15, crossover_p=0.9,
                 mutation_p=None, elite=2, tournament=3, seed=0, seeded=None):
        self.sizes = np.asarray(gene_sizes)
        self.fitness = fitness
        self.pop_size, self.generations = pop_size, generations
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p if mutation_p is not None else 1.0 / len(self.sizes)
        self.elite, self.tournament = elite, tournament
        self.rng = np.random.default_rng(seed)
        self.seeded = seeded or []     # known-good individuals to put in the first population
        self.evals = 0
        self._cache = {}

    def _score(self, genes):
        key = tuple(int(g) for g in genes)
        if key not in self._cache:
            self._cache[key] = float(self.fitness(np.array(key)))
            self.evals += 1
        return self._cache[key]

    def _random(self):
        return np.array([self.rng.integers(s) for s in self.sizes])

    def run(self, verbose=False):
        pop = [np.array(s) for s in self.seeded] + [self._random() for _ in range(self.pop_size - len(self.seeded))]
        history, mean_history = [], []
        best, best_f = None, np.inf
        for gen in range(self.generations):
            fit = np.array([self._score(p) for p in pop])
            order = np.argsort(fit)
            if fit[order[0]] < best_f:
                best, best_f = pop[order[0]].copy(), fit[order[0]]
            history.append(best_f)
            mean_history.append(float(fit.mean()))
            if verbose:
                print(f"  gen {gen + 1:2d}/{self.generations}  best {best_f:.4f}  mean {fit.mean():.4f}")
            nxt = [pop[i].copy() for i in order[: self.elite]]
            while len(nxt) < self.pop_size:
                a, b = self._pick(pop, fit), self._pick(pop, fit)
                child = a.copy()
                if self.rng.random() < self.crossover_p:        # uniform crossover
                    mask = self.rng.random(len(child)) < 0.5
                    child = np.where(mask, a, b)
                for i in range(len(child)):                       # mutation: reset the gene
                    if self.rng.random() < self.mutation_p:
                        child[i] = self.rng.integers(self.sizes[i])
                nxt.append(child)
            pop = nxt
        return best, best_f, {"best": [float(h) for h in history], "mean": mean_history, "evals": self.evals}

    def _pick(self, pop, fit):
        idx = self.rng.integers(len(pop), size=self.tournament)
        return pop[idx[np.argmin(fit[idx])]]


def binary_pso(n_bits, fitness, swarm=20, iters=15, w=0.7, c1=1.5, c2=1.5, vmax=4.0, seed=0, seeded=None):
    """Binary Particle Swarm Optimisation (sigmoid transfer). Minimises fitness(bits)."""
    rng = np.random.default_rng(seed)
    cache, evals = {}, 0

    def score(bits):
        nonlocal evals
        key = tuple(int(b) for b in bits)
        if key not in cache:
            cache[key] = float(fitness(np.array(key)))
            evals += 1
        return cache[key]

    pos = rng.integers(0, 2, size=(swarm, n_bits))
    for i, s in enumerate(seeded or []):
        pos[i] = s
    vel = rng.uniform(-1, 1, size=(swarm, n_bits))
    pbest = pos.copy()
    pbest_f = np.array([score(p) for p in pos])
    g = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[g].copy(), pbest_f[g]
    history = []
    for _ in range(iters):
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = np.clip(w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos), -vmax, vmax)
        pos = (rng.random(pos.shape) < 1.0 / (1.0 + np.exp(-vel))).astype(int)
        for i in range(swarm):
            f = score(pos[i])
            if f < pbest_f[i]:
                pbest[i], pbest_f[i] = pos[i].copy(), f
                if f < gbest_f:
                    gbest, gbest_f = pos[i].copy(), f
        history.append(float(gbest_f))
    return gbest, gbest_f, {"best": history, "evals": evals}
