"""
Count-based predictors of a haplotype's cluster in each haploblock, learned on the TRAINING
haplotypes of one chromosome; step 6 compares the fine-tuned model with them.

Tokens are block-local indices (0 .. K_b - 1 among the K_b tokens of haploblock b, see
GenomeData.block_pos); sequences are int arrays [haplotypes, blocks] in genomic order.

  frequency   P(x_b) = (n_b(x) + k) / (N + k K_b): ignores the haplotype; for haplotype units it is
              also the identity-free baseline (block + slot)
  bigram      add-k smoothed P(x_b | x_{b-1})
  trigram     P(x_b | x_{b-2}, x_{b-1}), backing off to the bigram and the frequency with a
              Dirichlet prior (hierarchical smoothing, strength 1)
  hmm         Li-Stephens copying model: the haplotype copies one training haplotype, switches to
              another with probability rho at every haploblock boundary, and differs from the copied
              cluster with probability eps; rho and eps are fitted on held-out training haplotypes.
              Predicts the next haploblock from the ones before (forward pass) and imputes masked
              haploblocks from both sides (forward-backward).
"""
import numpy as np
import scipy.sparse as sp

NEXT_BASELINES = ("frequency", "bigram", "trigram", "hmm")


class BlockBaselines:
    def __init__(self, train, K, k=0.5, rng=None, fit_haps=200):
        """train: int [N, B] block-local tokens of the training haplotypes; K: tokens per block [B]."""
        self.R, self.K, self.k = np.asarray(train), np.asarray(K), k
        N, B = self.R.shape
        self.freq = [np.bincount(self.R[:, b], minlength=self.K[b]).astype(np.float64) for b in range(B)]
        self.big = [self._counts(self.R[:, b - 1], self.R[:, b], self.K[b - 1], b) for b in range(1, B)]
        self.tri, self.ctx = [None, None], [None, None]
        for b in range(2, B):
            key, inv = np.unique(self.R[:, b - 2].astype(np.int64) * self.K[b - 1] + self.R[:, b - 1], return_inverse=True)
            self.ctx.append(key)
            self.tri.append(self._counts(inv, self.R[:, b], len(key), b))
        self.onehot = [sp.csr_matrix((np.ones(N), (np.arange(N), self.R[:, b])), shape=(N, self.K[b]))
                       for b in range(B)]
        self.rho, self.eps = self.fit_hmm(rng or np.random.default_rng(0), fit_haps)

    def _counts(self, a, b_tok, n_rows, b):
        return sp.csr_matrix((np.ones(len(a)), (a, b_tok)), shape=(n_rows, self.K[b]))

    # ---- next-haploblock prediction: per position b >= 1, a dense distribution over K_b

    def predict_next(self, Y, name):
        """Y: [n, B] test tokens -> list over b = 1..B-1 of [n, K_b] probabilities of x_b given x_<b."""
        if name == "hmm":
            return self.hmm(Y, causal=True)[1:]
        out = []
        for b in range(1, Y.shape[1]):
            p1 = (self.freq[b] + 1) / (self.freq[b].sum() + self.K[b])
            if name == "frequency":
                P = np.broadcast_to((self.freq[b] + self.k) / (self.freq[b].sum() + self.k * self.K[b]), (len(Y), self.K[b]))
            elif name == "bigram":
                n2 = self.big[b - 1][Y[:, b - 1]].toarray()
                P = (n2 + self.k) / (n2.sum(1, keepdims=True) + self.k * self.K[b])
            else:                                                # trigram -> bigram -> frequency
                n2 = self.big[b - 1][Y[:, b - 1]].toarray()
                p2 = (n2 + p1) / (n2.sum(1, keepdims=True) + 1)
                if b < 2:
                    P = p2
                else:
                    key = Y[:, b - 2].astype(np.int64) * self.K[b - 1] + Y[:, b - 1]
                    pos = np.searchsorted(self.ctx[b], key)
                    hit = (pos < len(self.ctx[b])) & (self.ctx[b][np.minimum(pos, len(self.ctx[b]) - 1)] == key)
                    n3 = np.zeros_like(p2)
                    n3[hit] = self.tri[b][pos[hit]].toarray()
                    P = (n3 + p2) / (n3.sum(1, keepdims=True) + 1)
            out.append(np.asarray(P))
        return out

    # ---- Li-Stephens HMM

    def hmm(self, Y, causal=True, mask=None, rho=None, eps=None, panel=None, batch=256):
        """Y: [n, B]. causal: P(x_b | x_<b) for every b (b = 0: the frequency). Otherwise mask [n, B]
        bool (True = unobserved) and, at the masked blocks, the posterior P(x_b | every observed block)
        by forward-backward. Returns a list over b of [n, K_b] probabilities."""
        rho = self.rho if rho is None else rho
        eps = self.eps if eps is None else eps
        R = self.R if panel is None else self.R[panel]
        oh = self.onehot if panel is None else [o[panel] for o in self.onehot]
        n, B = Y.shape
        out = [np.zeros((n, self.K[b])) for b in range(B)]
        for s in range(0, n, batch):
            y = Y[s:s + batch]
            m = None if mask is None else mask[s:s + batch]
            emit = lambda b: np.where(R[:, b][None, :] == y[:, b][:, None], 1 - eps, eps / self.K[b])
            seen = lambda b: np.ones(len(y), bool) if m is None else ~m[:, b]
            alpha = np.full((len(y), len(R)), 1.0 / len(R))
            fwd = []
            for b in range(B):
                if b:
                    alpha = (1 - rho) * alpha + rho / len(R)     # switch to a uniformly drawn haplotype
                if causal:
                    out[b][s:s + batch] = (1 - eps) * (alpha @ oh[b]) + eps / self.K[b]
                fwd.append(alpha)
                e = emit(b)
                alpha = np.where(seen(b)[:, None], alpha * e, alpha)
                alpha /= alpha.sum(1, keepdims=True)
            if causal:
                continue
            beta = np.ones_like(alpha)
            for b in range(B - 1, -1, -1):
                post = fwd[b] * beta
                post /= post.sum(1, keepdims=True)
                out[b][s:s + batch] = (1 - eps) * (post @ oh[b]) + eps / self.K[b]
                e = np.where(seen(b)[:, None], emit(b), 1.0)
                beta = (1 - rho) * (e * beta) + rho * (e * beta).mean(1, keepdims=True)
                beta /= beta.sum(1, keepdims=True)
        return out

    def fit_hmm(self, rng, n_fit):
        """rho, eps maximising the next-haploblock likelihood of up to n_fit training haplotypes,
        each predicted from a panel of the OTHER training haplotypes."""
        N, B = self.R.shape
        if N < 4 or B < 2:
            return 0.1, 0.01
        held = rng.choice(N, min(n_fit, N // 2), replace=False)
        panel = np.setdiff1d(np.arange(N), held)
        Y, best = self.R[held], (-np.inf, 0.1, 0.01)
        for rho in (0.01, 0.03, 0.1, 0.3):
            for eps in (0.001, 0.01, 0.05):
                P = self.hmm(Y, causal=True, rho=rho, eps=eps, panel=panel)
                ll = sum(np.log(P[b][np.arange(len(Y)), Y[:, b]]).sum() for b in range(1, B))
                if ll > best[0]:
                    best = (ll, rho, eps)
        return best[1], best[2]
