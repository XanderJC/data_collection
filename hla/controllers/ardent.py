from hla.controllers import BaseController
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class ArdentController(BaseController):
    def __init__(self, n_actions=10, n_explainers=5, alpha=0.5, key_root=41310):
        super().__init__(n_actions, n_explainers)

        self.explainers = list(range(n_explainers))
        self.key = jax.random.PRNGKey(key_root)

        vars = dict()
        vars["N"] = 2000
        vars["E"] = n_explainers
        vars["X"] = 1
        vars["A"] = 1

        vars["hyp_alp"] = alpha
        vars["hyp_bet"] = jnp.sqrt(1 - vars["hyp_alp"] ** 2)
        vars["pi_source"] = jnp.array([1.0])

        self.vars = vars

        self.key, subkey = jax.random.split(self.key)

        self.qs = jax.random.normal(
            subkey, shape=(vars["N"], vars["E"], vars["X"], vars["A"])
        )
        self.ws = jnp.ones(vars["N"]) / vars["N"]

    def select_explainers(self):

        self.key, subkey = jax.random.split(self.key)
        x = 0
        k = jax.random.choice(
            subkey, jnp.arange(self.vars["N"]), p=self.ws
        )  # posterior sample particle
        _bs_log = jnp.log(self.vars["pi_source"][x])[None, ...] + self.qs[k, :, x, :]
        _bs_log = _bs_log - logsumexp(_bs_log, keepdims=True)
        explainers = jnp.argsort(_bs_log[:, 0])
        explainers = jnp.flip(explainers, 0)
        e = jnp.argmax(_bs_log[:, 0])

        return explainers

    def _posterior_update(self, qs, ws, explainers):

        N, E, X, A = qs.shape
        x = 0
        e = explainers  # list of explainers seen e.g. [2,4,3,0]
        a = 0

        # posterior update:

        q_bar = jnp.einsum("n,nexa->exa", ws, qs)
        q_var = jnp.einsum(
            "n,nexa,nEXA->exaEXA", ws, qs - q_bar[None, ...], qs - q_bar[None, ...]
        )
        _ms = self.vars["hyp_alp"] * qs + (1 - self.vars["hyp_alp"]) * q_bar[None, ...]

        _bs_log = jnp.log(self.vars["pi_source"][x])
        for i, e in enumerate(explainers):
            _bs_log += (i + 1) * _ms[:, e, x]

        _bs_log = _bs_log - logsumexp(_bs_log, keepdims=True)

        _ps = jnp.log(ws) + _bs_log[:, a]
        _ps = jnp.exp(_ps - logsumexp(_ps))

        self.key, subkey = jax.random.split(self.key)
        _ks = jax.random.choice(subkey, jnp.arange(N), p=_ps, shape=(N,))
        _ms = _ms[_ks, ...]

        self.key, subkey = jax.random.split(self.key)
        _means = _ms.reshape(N, E * X * A)
        _cov_u, _cov_s, _ = jnp.linalg.svd(q_var.reshape(E * X * A, E * X * A))
        _cov_root = self.vars["hyp_bet"] * (_cov_u @ jnp.diag(jnp.sqrt(_cov_s)))
        qs = _means + jnp.einsum(
            "dD,nD->nd", _cov_root, jax.random.normal(subkey, shape=(N, E * X * A))
        )
        qs = qs.reshape(N, E, X, A)

        _bqs_log = jnp.log(self.vars["pi_source"][x])[None, ...]
        _bms_log = jnp.log(self.vars["pi_source"][x])[None, ...]

        for i, e in enumerate(explainers):
            _bqs_log += (i + 1) * qs[:, e, x]
            _bms_log += (i + 1) * _ms[_ks, ...][:, e, x]


        _bqs_log = _bqs_log - logsumexp(_bqs_log, keepdims=True)

        _bms_log = _bms_log - logsumexp(_bms_log, keepdims=True)
        ws = _bqs_log[:, a] - _bms_log[:, a]
        ws = jnp.exp(ws - logsumexp(ws))

        return qs, ws

    def update(
        self,
        init_action,
        updated_action,
        explanations_viewed,
        example_index,
        explainers_given,
    ):

        explainers = explainers_given[:explanations_viewed]

        if (
            (explanations_viewed != 0)
            & (init_action != updated_action)
            & (updated_action != -1)
        ):
            qs, ws = self._posterior_update(self.qs, self.ws, explainers)
        else:
            qs, ws = self.qs, self.ws

        self.state = (qs, ws)
        self.qs, self.ws = self.state

        self.memory.append(
            (
                init_action,
                updated_action,
                explanations_viewed,
                example_index,
                explainers_given,
                # copy.deepcopy(self.state),
                self.state,
            )
        )

        return
