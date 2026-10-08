"""Small NumPy reference operations. Axes and rejection behavior are explicit."""
import math
import numpy as np


def softmax(logits, axis=-1):
    x = np.asarray(logits, dtype=np.float64)
    if x.size == 0 or np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError("logits must be nonempty and contain no NaN or +infinity")
    maximum = x.max(axis=axis, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("each row needs at least one finite logit")
    weights = np.exp(x - maximum)
    return weights / weights.sum(axis=axis, keepdims=True)


def rms_norm(x, scale, epsilon=1e-6):
    x, scale = np.asarray(x, dtype=float), np.asarray(scale, dtype=float)
    if x.ndim < 1 or x.shape[-1] < 1 or scale.shape != (x.shape[-1],) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("expected [..., width], [width], and positive epsilon")
    if not np.isfinite(x).all() or not np.isfinite(scale).all():
        raise ValueError("normalization inputs must be finite")
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + epsilon) * scale


def rope(x, positions, base=10000.0):
    """Adjacent-pair rotation for [tokens, even head_width]; no scaling extension."""
    x, positions = np.asarray(x, dtype=float), np.asarray(positions, dtype=float)
    if x.ndim != 2 or x.shape[-1] % 2 or positions.shape != (len(x),):
        raise ValueError("expected [tokens, even width] and one position per token")
    if base <= 0 or not np.isfinite(base) or not np.isfinite(x).all() or not np.isfinite(positions).all():
        raise ValueError("finite inputs and positive base required")
    angles = positions[:, None] / (base ** (np.arange(0, x.shape[1], 2) / x.shape[1]))
    even, odd = x[:, 0::2], x[:, 1::2]
    result = np.empty_like(x)
    result[:, 0::2] = even*np.cos(angles) - odd*np.sin(angles)
    result[:, 1::2] = even*np.sin(angles) + odd*np.cos(angles)
    return result


def attention(q, k, v, allowed=None):
    """Single-head reference: [queries,d], [keys,d], [keys,value_width]."""
    q, k, v = (np.asarray(a, dtype=float) for a in (q, k, v))
    if any(a.ndim != 2 for a in (q, k, v)) or q.shape[1] != k.shape[1] or len(k) != len(v) or q.shape[1] < 1:
        raise ValueError("incompatible query, key, or value axes")
    if not all(np.isfinite(a).all() for a in (q, k, v)):
        raise ValueError("attention inputs must be finite")
    scores = q @ k.T / math.sqrt(q.shape[1])
    if allowed is not None:
        allowed = np.asarray(allowed)
        if allowed.dtype != np.bool_ or allowed.shape != scores.shape:
            raise ValueError("allowed must be a boolean [queries,keys] mask")
        scores = np.where(allowed, scores, -np.inf)
    weights = softmax(scores)
    return weights @ v, weights


def token_loss(logits, targets, supervised=None):
    """Mean next-token NLL over selected positions; caller supplies shifted targets."""
    logits, targets = np.asarray(logits, dtype=float), np.asarray(targets)
    if logits.ndim != 3 or targets.shape != logits.shape[:-1] or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("expected logits [batch,time,vocab] and integer targets [batch,time]")
    if (targets < 0).any() or (targets >= logits.shape[-1]).any() or not np.isfinite(logits).all():
        raise ValueError("invalid target or nonfinite logit")
    mask = np.ones_like(targets, dtype=bool) if supervised is None else np.asarray(supervised)
    if mask.shape != targets.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("need a boolean mask with at least one supervised position")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=-1))
    chosen = np.take_along_axis(shifted, targets[..., None], axis=-1)[..., 0]
    return float((log_normalizer-chosen)[mask].mean()), int(mask.sum())


def quantize_symmetric(values, bits=4):
    """One signed symmetric group; returns codes, scale, reconstructed float array."""
    x = np.asarray(values, dtype=float)
    if type(bits) is not int or not 2 <= bits <= 16 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("need finite nonempty values and integer bits in [2,16]")
    bound = 2 ** (bits-1)-1
    maximum = float(np.abs(x).max())
    scale = maximum/bound if maximum else 1.0
    codes = np.clip(np.rint(x/scale), -bound, bound).astype(np.int32)
    return codes, scale, codes*scale


def run_checks():
    results = []
    def check(name, condition, observed, explanation):
        if not bool(condition):
            raise AssertionError(name)
        results.append({"id": name, "passed": True, "observed": observed, "interpretation": explanation})
    x = np.array([1000., 1001., 999.])
    p = softmax(x)
    check("softmax_shift", np.allclose(p, softmax(x-1000)), p.tolist(), "Adding a common logit offset does not change probabilities.")
    q, k, v = np.array([[1., 0.]]), np.eye(2), np.array([[10.], [2.]])
    out, weights = attention(q,k,v)
    check("attention_read", np.isclose(out.item(),7.358092394613255), {"output":out.item(),"weights":weights.tolist()}, "Keys determine addressing weights; values determine delivered content.")
    rng = np.random.default_rng(13)
    q, k, v = [rng.normal(size=(4,4)) for _ in range(3)]
    mask = np.tril(np.ones((4,4),dtype=bool))
    original,_ = attention(q,k,v,mask)
    changed = v.copy(); changed[-1] += 1000
    altered,_ = attention(q,k,changed,mask)
    check("causal_noninterference", np.allclose(original[:-1],altered[:-1]), float(np.max(np.abs(original[:-1]-altered[:-1]))), "Changing a future value cannot change earlier outputs.")
    rotated = rope(q, np.arange(4))
    check("rope_norm", np.allclose(np.linalg.norm(q,axis=1),np.linalg.norm(rotated,axis=1)), float(np.max(np.abs(np.linalg.norm(q,axis=1)-np.linalg.norm(rotated,axis=1)))), "Pairwise rotations preserve length, not semantic quality.")
    normalized = rms_norm(np.array([[3.,4.]]),np.ones(2))
    check("rms_norm", np.allclose(normalized, [[3/math.sqrt(12.5+1e-6),4/math.sqrt(12.5+1e-6)]]), normalized.tolist(), "RMS normalization rescales without subtracting the mean.")
    inputs,w,a,b = [rng.normal(size=s) for s in [(3,4),(4,5),(4,2),(2,5)]]
    direct = inputs@w + .5*(inputs@a)@b
    merged = inputs@(w+.5*a@b)
    check("lora_merge", np.allclose(direct,merged), float(np.max(np.abs(direct-merged))), "Floating-point low-rank merge is algebraically equivalent at the same scale.")
    codes,scale,reconstruction=quantize_symmetric([-.8,-.1,.2,1.0])
    error=float(np.max(np.abs(reconstruction-np.array([-.8,-.1,.2,1.]))))
    check("quantization_error", error <= scale/2+1e-12, {"codes":codes.tolist(),"scale":scale,"maximum_absolute_error":error}, "In-range round-to-nearest error is bounded by half a quantization step; this is not a packed kernel.")
    logits=np.array([[[3.,0.],[0.,3.],[100.,-100.]]])
    loss,count=token_loss(logits,np.array([[0,1,1]]),np.array([[True,True,False]]))
    check("loss_denominator", count==2 and np.isclose(loss,np.log1p(np.exp(-3))), {"loss":loss,"supervised_tokens":count}, "Unsupervised positions contribute neither numerator nor denominator.")
    return {"result_class":"numerical-reference-checks","passed":True,"total":len(results),"checks":results}
