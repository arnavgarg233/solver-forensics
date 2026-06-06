"""
solver-forensics - identify the discretization scheme behind a numerical solver
from its output, via the truncation-error (modified-equation) signature.

Quick start
-----------
    import numpy as np, solver_forensics as sf

    # one field -> its modified-equation fingerprint (unit coefficient direction)
    fp = sf.signature(u_solver, u_ref, dx=L/N)

    # attribute a scheme across many initial conditions, with a permutation floor
    result = sf.audit({
        "scheme_A": [(uA[i], uref[i]) for i in range(n_ic)],
        "scheme_B": [(uB[i], uref[i]) for i in range(n_ic)],
    }, dx=L/N)
    # -> {'accuracy': 0.99, 'permutation_floor': 0.50, 'margin': 0.49, 'admissible': True}

The signature is the unit-normalized direction of ``c`` in
``r = u_solver - u_ref ~= sum_p c_p d^p u / dx^p``, recovered by least squares against a small
derivative library. Attribution is a GroupKFold-by-initial-condition classification scored
against a label-permutation floor - the same controls the paper uses, so a positive call comes
with its own admissibility check.

Reference: "A modified-equation signature method for identifying hidden discretization schemes."
"""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

__version__ = "1.0.0"
__all__ = ["signature", "attribute", "audit", "DEFAULT_LIBRARY"]

DEFAULT_LIBRARY = (2, 3, 4)   # {u_xx, u_xxx, u_xxxx}: numerical diffusion + dispersion terms


def _deriv(u, p, dx):
    """Central, periodic finite-difference d^p u / dx^p (O(dx^2)) along the last axis."""
    r = lambda s: np.roll(u, s, axis=-1)
    if p == 2:
        return (r(-1) - 2 * u + r(1)) / dx ** 2
    if p == 3:
        return (r(-2) - 2 * r(-1) + 2 * r(1) - r(2)) / (2 * dx ** 3)
    if p == 4:
        return (r(-2) - 4 * r(-1) + 6 * u - 4 * r(1) + r(2)) / dx ** 4
    if p == 5:
        return (r(-3) - 4 * r(-2) + 5 * r(-1) - 5 * r(1) + 4 * r(2) - r(3)) / (2 * dx ** 5)
    raise ValueError(f"derivative order {p} not supported (use 2-5)")


def signature(u_solver, u_ref, dx=1.0, library=DEFAULT_LIBRARY):
    """Modified-equation signature: the unit-normalized truncation-error coefficient direction.

    Parameters
    ----------
    u_solver, u_ref : array_like
        Equal-shape, periodic fields on a uniform grid of spacing ``dx``. Either 1D ``(N,)`` for
        a single field, or 2D ``(M, N)`` for M fields (e.g. one per initial condition).
    dx : float
        Grid spacing.
    library : tuple of int
        Derivative orders to fit (default ``(2, 3, 4)`` = diffusion + dispersion).

    Returns
    -------
    ndarray
        Unit coefficient direction - shape ``(len(library),)`` for 1D input, ``(M, len(library))``
        for 2D. This is the magnitude-invariant fingerprint of the scheme.
    """
    u = np.asarray(u_solver, float)
    one_d = u.ndim == 1
    U = np.atleast_2d(u)
    R = U - np.atleast_2d(np.asarray(u_ref, float))           # residual r = u_solver - u_ref
    A = np.stack([_deriv(U, p, dx) for p in library], axis=2)  # (M, N, P)
    P = len(library)
    AtA = np.einsum("mni,mnk->mik", A, A) + 1e-8 * np.eye(P)
    Atb = np.einsum("mni,mn->mi", A, R)
    c = np.linalg.solve(AtA, Atb[..., None])[..., 0]          # (M, P) coefficients
    unit = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    return unit[0] if one_d else unit


def attribute(signatures, labels, groups, n_splits=5, n_permutations=100,
              admissible_margin=0.15, random_state=0):
    """Cross-validated scheme attribution with a label-permutation floor.

    Parameters
    ----------
    signatures : array_like, shape (M, P)
        Signatures from :func:`signature`.
    labels : array_like, shape (M,)
        Scheme label for each signature.
    groups : array_like, shape (M,)
        Initial-condition / run id for each signature. GroupKFold keeps each id out of the fold
        it is tested in, so no score reflects memorizing a single field.
    admissible_margin : float
        A call is reported admissible if accuracy exceeds the permutation floor by this margin.

    Returns
    -------
    dict
        ``{'accuracy', 'permutation_floor', 'margin', 'admissible'}``.
    """
    X = np.asarray(signatures, float)
    y = np.asarray(labels)
    g = np.asarray(groups)
    k = min(n_splits, len(np.unique(g)))
    if k < 2:
        raise ValueError("need >= 2 distinct groups for GroupKFold-by-initial-condition")
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = GroupKFold(k)
    acc = float(cross_val_score(mk(), X, y, groups=g, cv=cv).mean())
    rng = np.random.default_rng(random_state)
    floor = float(np.median([
        cross_val_score(mk(), X, rng.permutation(y), groups=g, cv=cv).mean()
        for _ in range(n_permutations)
    ]))
    margin = acc - floor
    return {"accuracy": acc, "permutation_floor": floor,
            "margin": margin, "admissible": bool(margin > admissible_margin)}


def audit(samples, dx=1.0, library=DEFAULT_LIBRARY, **kwargs):
    """High-level audit over multiple schemes and initial conditions.

    Parameters
    ----------
    samples : dict
        ``{scheme_label: [(u_solver, u_ref), ...]}`` - one ``(solver, reference)`` field pair per
        initial condition. Pairs are aligned across schemes by position, so position = IC id =
        GroupKFold group.
    dx, library : passed to :func:`signature`.
    **kwargs : passed to :func:`attribute` (e.g. ``n_permutations``, ``admissible_margin``).

    Returns
    -------
    dict
        The :func:`attribute` result.
    """
    sigs, labels, groups = [], [], []
    for label, pairs in samples.items():
        for ic, (us, ur) in enumerate(pairs):
            sigs.append(signature(us, ur, dx=dx, library=library))
            labels.append(label)
            groups.append(ic)
    return attribute(np.asarray(sigs), np.asarray(labels), np.asarray(groups), **kwargs)
