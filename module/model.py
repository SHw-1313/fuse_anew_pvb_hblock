import torch
from torch import nn
import numpy as np
import math
from copy import deepcopy
from .torchmd_et import TorchMD_VQ_ET, ParamGVPFFNLayer, _init_linear_
# from .equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20 as EquiformerV2
from .interpolant_matcher import INTERP_MATCHER
from .graph import construct_edges
import pdb

from utils import *
from utils.geometry import random_rotation_matrix, compute_crmsd_torch


def compute_grad(v, x, grad_outputs=None, create_graph=False):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(v)
    grad_v_to_x = torch.autograd.grad(
        outputs=v,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        allow_unused=True
    )[0]
    return grad_v_to_x


class dyVAE(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, rbf_dim, heads, layers,
                 sphere_channels=128,
                 attn_hidden_channels=64,
                 attn_alpha_channels=64,
                 attn_value_channels=16,
                 cutoff_lower=0.0,
                 cutoff_upper=5.0,
                 cutoff_H=3.5,
                 k_neighbors=16,
                 coord_prior_var=1.0,
                 sigma=0.2,
                 additional_noise_scale=0.2,
                 kl_weight=0.5,
                 re_weight=0.5,
                 using_ode=True,
                 backbone="torchmdnet",
                 fusion_mode="off",
                 anew_encoder_config=None,
                 shared_hblock_variant="real",
                 shared_hblock_seed=20260810
                ):
        super(dyVAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.rbf_dim = rbf_dim
        self.heads = heads
        self.layers = layers
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.cutoff_H = cutoff_H
        self.k_neighbors = k_neighbors
        self.coord_prior_var = coord_prior_var
        self.sigma = sigma
        self.additional_noise_scale = additional_noise_scale
        self.kl_weight = kl_weight
        self.re_weight = re_weight
        self.using_ode = using_ode
        self.backbone = backbone
        if fusion_mode not in {"off", "anew_block", "anew_block_pvb_posterior", "pvb_shared_hblock"}:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}; expected 'off', "
                "'anew_block', 'anew_block_pvb_posterior', or 'pvb_shared_hblock'"
            )
        self.fusion_mode = fusion_mode

        self.anew_block_encoder = None
        self.shared_hblock_variant = str(shared_hblock_variant)
        self.shared_hblock_seed = int(shared_hblock_seed)
        self.block_projection = None
        self.block_gate = None
        self.shared_hblock_adapter = None
        self.shared_hblock_gate = None
        self.anew_hidden_dim = None
        if self.fusion_mode in {"anew_block", "anew_block_pvb_posterior"}:
            from .anew_block_encoder import AnewBlockEncoder

            encoder_config = dict(anew_encoder_config or {})
            encoder_config.setdefault("hidden_size", 512)
            self.anew_hidden_dim = int(encoder_config["hidden_size"])
            self.anew_block_encoder = AnewBlockEncoder(**encoder_config)
            self.block_projection = nn.Sequential(
                nn.LayerNorm(self.anew_hidden_dim),
                nn.Linear(self.anew_hidden_dim, hidden_dim),
            )
            self.block_gate = nn.Parameter(torch.zeros(1))
        elif self.fusion_mode == "pvb_shared_hblock":
            from .shared_hblock import SharedHBlockAdapter

            self.shared_hblock_adapter = SharedHBlockAdapter(
                hidden_dim,
                rank=32,
                variant=self.shared_hblock_variant,
                shuffle_seed=self.shared_hblock_seed,
            )
            self.shared_hblock_gate = nn.Parameter(torch.zeros(1))

        # encoder
        self.encoder = TorchMD_VQ_ET(
            hidden_channels=hidden_dim, extra_channels=0, num_layers=2,
            num_rbf=rbf_dim, num_heads=heads, cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper,
            max_z=NUM_ATOM_TYPE, max_b=NUM_BLOCK_TYPE, cross_attn=False
        )
        self.W_vec_mu = nn.Linear(hidden_dim, 1, bias=False)
        self.W_vec_log_var = nn.Linear(hidden_dim, 3)

        # decoder
        if self.backbone == "torchmdnet":
            self.decoder = TorchMD_VQ_ET(
                hidden_channels=hidden_dim, extra_channels=1, num_layers=layers,
                num_rbf=rbf_dim, num_heads=heads, cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper,
                max_z=NUM_ATOM_TYPE, max_b=NUM_BLOCK_TYPE, cross_attn=True
            )
        elif self.backbone == "equiformer-v2":
            # NOTE: have not implemented cross-attention version yet !!!
            self.decoder = EquiformerV2(
                max_num_elements=NUM_ATOM_TYPE, max_num_blocks=NUM_BLOCK_TYPE,
                num_layers=layers, sphere_channels=sphere_channels, attn_hidden_channels=attn_hidden_channels,
                num_heads=heads, attn_alpha_channels=attn_alpha_channels, attn_value_channels=attn_value_channels,
                ffn_hidden_channels=ffn_dim, output_channels=hidden_dim,
                norm_type="layer_norm_sh", lmax_list=[2], mmax_list=[2], grid_resolution=None,
                num_sphere_samples=128,
                edge_channels=rbf_dim, use_atom_edge_embedding=True, share_atom_edge_embedding=False,
                attn_activation="silu", use_s2_act_attn=False, use_attn_renorm=True,
                ffn_activation="silu", use_gate_act=False, use_grid_mlp=True, use_sep_s2_act=True,
                alpha_drop=0.1, drop_path_rate=0.1, proj_drop=0, weight_init="uniform"
            )
        else:
            raise NotImplementedError(f"Backbone model {self.backbone} not implemented yet.")
        self.vel_ffn = ParamGVPFFNLayer(hidden_dim, ffn_dim, vector_norm=True)
        if not self.using_ode:
            self.drf_ffn = ParamGVPFFNLayer(hidden_dim, ffn_dim, vector_norm=True)

        _init_linear_(self.W_vec_mu)
        _init_linear_(self.W_vec_log_var)
    
    def encode(self, z, b, x, x_mu_ref, batch, edge_index, edge_weight, edge_vec, bond_type, mask, return_state=False):
        """Encode PVB coordinates and optionally expose scalar atom state.

        The default two-value return is kept for all historical callers.  The
        optional state is used by the Phase 11 shared-H-block path and does not
        change the posterior calculation.
        """
        h, vec, _, _, _ = self.encoder(z=z, b=b, pos=x, batch=batch, edge_index=edge_index,
                                       edge_weight_t=edge_weight, edge_vec_t=edge_vec,
                                       bond_type=bond_type)
        # preserve SE(3) equivariance
        # x_mu = self.W_vec_mu(vec).squeeze() + x
        x_mu = x.clone()
        # kl loss
        # x_mu_delta = x_mu - x_mu_ref
        x_log_var = -torch.abs(self.W_vec_log_var(h))
        # kl_loss = -0.5 * torch.sum(1.0 + x_log_var[mask] - math.log(self.coord_prior_var) - x_mu_delta[mask] * x_mu_delta[mask] / self.coord_prior_var - torch.exp(x_log_var[mask]) / self.coord_prior_var) / mask.sum()
        kl_loss = -0.5 * torch.sum(1.0 + x_log_var[mask] - math.log(self.coord_prior_var) - torch.exp(x_log_var[mask]) / self.coord_prior_var) / mask.sum()
        # reparametrization
        x_rep = x_mu + torch.exp(x_log_var / 2) * torch.randn_like(x)
        if return_state:
            return x_rep, kl_loss, {
                "h_atom": h,
                "vec_atom": vec,
                "log_var_pvb": x_log_var,
            }
        return x_rep, kl_loss

    def _encode_pvb_path(self, z, b, x, x_mu_ref, abid, mask, edge_mask, bond_index, return_state=False):
        """Run the original PVB encoder/graph path used by ``off``."""

        e_edge_index, e_edge_weight, e_edge_vec, e_bond_type = construct_edges(
            Z=z,
            X=x,
            bid=abid,
            mask=edge_mask,
            bond_index=bond_index,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            cutoff_H=self.cutoff_H,
            k_neighbors=self.k_neighbors,
        )
        return self.encode(
            z,
            b,
            x,
            x_mu_ref,
            abid,
            e_edge_index,
            e_edge_weight,
            e_edge_vec,
            e_bond_type,
            mask,
            return_state=return_state,
        )

    def _pvb_encoder_diagnostics(self, z, b, x, abid, mask, edge_mask, bond_index):
        """Return PVB encoder state and posterior variance without sampling."""

        e_edge_index, e_edge_weight, e_edge_vec, e_bond_type = construct_edges(
            Z=z,
            X=x,
            bid=abid,
            mask=edge_mask,
            bond_index=bond_index,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            cutoff_H=self.cutoff_H,
            k_neighbors=self.k_neighbors,
        )
        h, vec, _, _, _ = self.encoder(
            z=z,
            b=b,
            pos=x,
            batch=abid,
            edge_index=e_edge_index,
            edge_weight_t=e_edge_weight,
            edge_vec_t=e_edge_vec,
            bond_type=e_bond_type,
        )
        log_var = -torch.abs(self.W_vec_log_var(h))
        denominator = mask.sum().clamp_min(1).to(dtype=x.dtype)
        kl_loss = -0.5 * torch.sum(
            1.0
            + log_var[mask]
            - math.log(self.coord_prior_var)
            - torch.exp(log_var[mask]) / self.coord_prior_var
        ) / denominator
        return {
            "H_pvb": h,
            "log_var_pvb": log_var,
            "kl_loss_pvb": kl_loss,
        }

    def _run_anew_block_encoder(self, x, batch):
        if self.anew_block_encoder is None:
            raise RuntimeError("Anew block encoder is not constructed for this mode")
        return self.anew_block_encoder(
            x_atom=x,
            atom_type=batch["atype"],
            block_type=batch["block_type"],
            atom_block_id=batch["atom_block_id"],
            block_batch=batch["block_batch"],
            block_lengths=batch["block_lengths"],
            bond_index=batch.get("bond_index"),
            bond_type=batch.get("bond_type"),
        )

    def _encode_anew_block(self, x, batch, mask):
        """Legacy Phase 9 Anew-posterior path; keep its semantics unchanged."""

        if self.fusion_mode != "anew_block" or self.anew_block_encoder is None:
            raise RuntimeError("_encode_anew_block requires fusion_mode='anew_block'")
        block_output = self._run_anew_block_encoder(x, batch)

        # Milestone-one coordinate contract: Anew coordinates are diagnostic;
        # the bridge source mean remains exactly the PVB input structure.
        x_mu = x.clone()
        atom_log_var = -torch.abs(
            block_output["log_var_block"].index_select(
                0, block_output["atom_block_id"]
            )
        ).expand(-1, x.shape[-1])
        denominator = mask.sum().clamp_min(1).to(dtype=x.dtype)
        kl_loss = -0.5 * torch.sum(
            1.0
            + atom_log_var[mask]
            - math.log(self.coord_prior_var)
            - torch.exp(atom_log_var[mask]) / self.coord_prior_var
        ) / denominator
        x_rep = x_mu + torch.exp(atom_log_var / 2) * torch.randn_like(x)
        return x_rep, kl_loss, block_output

    def _encode_anew_conditioning(self, x, batch):
        """Return Anew diagnostics for H-block decoder conditioning only."""

        if self.fusion_mode not in {"anew_block", "anew_block_pvb_posterior"}:
            raise RuntimeError(
                "_encode_anew_conditioning requires an Anew block fusion mode"
            )
        return self._run_anew_block_encoder(x, batch)

    def _project_block_condition(self, block_output):
        if self.fusion_mode not in {"anew_block", "anew_block_pvb_posterior"}:
            return None
        condition = self.block_projection(block_output["H_block"])
        condition = condition.index_select(0, block_output["atom_block_id"])
        # Keep the gate scalar explicit and shared by both decoder branches.
        return torch.tanh(self.block_gate) * condition

    def _project_shared_hblock_condition(self, h_atom, batch):
        """Pool detached PVB features and return the gated atom condition."""

        if self.fusion_mode != "pvb_shared_hblock" or self.shared_hblock_adapter is None:
            raise RuntimeError(
                "_project_shared_hblock_condition requires fusion_mode='pvb_shared_hblock'"
            )
        output = self.shared_hblock_adapter(
            h_atom=h_atom,
            atom_block_id=batch["atom_block_id"],
            block_lengths=batch["block_lengths"],
            block_batch=batch.get("block_batch"),
            variant=self.shared_hblock_variant,
        )
        condition_atom = torch.tanh(self.shared_hblock_gate) * output["condition_atom"]
        return condition_atom, output

    def decode(self, z, b, x, t, batch, edge_index, edge_weight_0, edge_vec_0, edge_weight_t, edge_vec_t, bond_type, block_condition=None, post_cross_condition=None):
        if self.backbone == "torchmdnet":
            h, vec, _, _, _ = self.decoder(z=z, b=b, pos=x, batch=batch, t=t,edge_index=edge_index,
                                           edge_weight_0=edge_weight_0, edge_vec_0=edge_vec_0,
                                           edge_weight_t=edge_weight_t, edge_vec_t=edge_vec_t,
                                           bond_type=bond_type, block_condition=block_condition,
                                           post_cross_condition=post_cross_condition)
        elif self.backbone == "equiformer-v2":
            if block_condition is not None or post_cross_condition is not None:
                raise NotImplementedError(
                    "Anew block conditioning is only implemented for torchmdnet"
                )
            h, vec = self.decoder(z=z, b=b, pos=x, batch=batch, t=t, edge_index=edge_index,
                                  edge_distance=edge_weight_t, edge_distance_vec=edge_vec_t,
                                  bond_type=bond_type)
        else:
            raise NotImplementedError(f"Backbone model {self.backbone} not implemented yet.")
        _, vel_out = self.vel_ffn(h, vec)
        vel_pred = torch.sum(vel_out, dim=-1)  # (N, 3)
        if not self.using_ode:
            _, drf_out = self.drf_ffn(h, vec)
            drf_pred = torch.sum(drf_out, dim=-1)
            return vel_pred, drf_pred
        return vel_pred

    def _train(self, batch, mode="pretrain"):
        x0, b0, atype, btype, abid, mask, edge_mask, bond_index = batch["x0"], batch["b0"], batch["atype"], batch["btype"], batch["abid"], batch["mask"], batch["edge_mask"], batch["bond_index"]
        if mode == "md":
            x1 = batch["x1"]
        # the prior of latent variables is a Gaussian centered at the original positions for each heavy atom of x0
        b0 = x0.clone()

        clean_x0 = x0.clone()
        if self.training:
            noise = torch.randn_like(x0) * self.additional_noise_scale
            x0 = x0 + noise
        
        target = x1 if mode == "md" else clean_x0

        post_cross_condition = None
        pvb_state = None
        N = x0.shape[0]
        # encode
        block_condition = None
        if self.fusion_mode in {"off", "anew_block_pvb_posterior", "pvb_shared_hblock"}:
            # Preserve the original PVB encoder path for the baseline,
            # corrected posterior-preserving mode, and shared-PVB Phase 11 mode.
            if self.fusion_mode == "pvb_shared_hblock":
                x_rep, kl_loss, pvb_state = self._encode_pvb_path(
                    atype, btype, x0, b0, abid, mask, edge_mask, bond_index,
                    return_state=True,
                )
            else:
                x_rep, kl_loss = self._encode_pvb_path(
                    atype, btype, x0, b0, abid, mask, edge_mask, bond_index
                )
            if self.fusion_mode == "anew_block_pvb_posterior":
                # Anew consumes the clean input and supplies decoder-only block
                # conditioning; its posterior/coordinate outputs are ignored
                # by the bridge objective.
                block_output = self._encode_anew_conditioning(clean_x0, batch)
                block_condition = self._project_block_condition(block_output)
            elif self.fusion_mode == "pvb_shared_hblock":
                post_cross_condition, _ = self._project_shared_hblock_condition(
                    pvb_state["h_atom"], batch
                )
        else:
            # Legacy Phase 9 semantics: Anew supplies both the posterior and
            # block conditioning. Keep this branch unchanged in meaning.
            x_rep, kl_loss, block_output = self._encode_anew_block(
                clean_x0, batch, mask
            )
            block_condition = self._project_block_condition(block_output)

        # in case of invalid data points
        if torch.any(torch.isnan(x_rep)):
            # pdb.set_trace()
            nan_loss = torch.tensor(torch.nan, device=x0.device)
            return nan_loss, (nan_loss, nan_loss, nan_loss)
        
        # decode: bridge matching
        t_all_one = torch.ones(N, 1, dtype=x0.dtype, device=x0.device)
        # uniformly sample time step
        t = np.random.rand()  # use the same time step for all mini-batches
        t_diff = t_all_one * t

        # sample interpolants
        xt, vel_gt, den_gt = INTERP_MATCHER.sample_conditional_flow(x_rep, target, t_diff, self.sigma)
        velocity_gt = INTERP_MATCHER.velocity(vel_gt, den_gt, t_diff, self.sigma)
        drf_gt = INTERP_MATCHER.drift(vel_gt, den_gt, t_diff, self.sigma)

        # construct edges
        d_edge_index, d_edge_weight_t, d_edge_vec_t, d_bond_type = construct_edges(
            Z=atype, X=xt, bid=abid, mask=edge_mask, bond_index=bond_index,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            cutoff_H=self.cutoff_H,
            k_neighbors=self.k_neighbors
        )

        d_edge_atm_0 = x_rep[d_edge_index.transpose(0, 1)]      # (Ef, 2, 3)
        d_edge_vec_0 = d_edge_atm_0[:, 0] - d_edge_atm_0[:, 1]  # (Ef, 3)
        d_edge_weight_0 = torch.norm(d_edge_vec_0, dim=-1)      # (Ef,)

        lig_mask = (edge_mask == 1)

        if self.using_ode:
            # NOTE: not theoretically proved to preserve coupling information !!!
            vel_pred = self.decode(atype, btype, xt, t_diff, abid, d_edge_index,
                                   d_edge_weight_0, d_edge_vec_0,
                                   d_edge_weight_t, d_edge_vec_t,
                                   d_bond_type, block_condition=block_condition,
                                   post_cross_condition=post_cross_condition)
            rec_vel_loss = F.mse_loss(vel_pred[mask], velocity_gt[mask], reduction='mean')
            rec_drf_loss = 0.
            if torch.any(lig_mask):
                rec_vel_loss += F.mse_loss(vel_pred[lig_mask], velocity_gt[lig_mask], reduction='mean')
        else:
            vel_pred, drf_pred = self.decode(atype, btype, xt, t_diff, abid, d_edge_index,
                                             d_edge_weight_0, d_edge_vec_0,
                                             d_edge_weight_t, d_edge_vec_t,
                                             d_bond_type, block_condition=block_condition,
                                             post_cross_condition=post_cross_condition)
            rec_vel_loss = F.mse_loss(vel_pred[mask], velocity_gt[mask], reduction='mean')
            rec_drf_loss = F.mse_loss(drf_pred[mask], drf_gt[mask], reduction='mean')
            if torch.any(lig_mask):
                rec_vel_loss += F.mse_loss(vel_pred[lig_mask], velocity_gt[lig_mask], reduction='mean')
                rec_drf_loss += F.mse_loss(drf_pred[lig_mask], drf_gt[lig_mask], reduction='mean')

        loss = self.kl_weight * kl_loss + self.re_weight * (rec_vel_loss + rec_drf_loss)
        return loss, (kl_loss, rec_vel_loss, rec_drf_loss)

    
    def inference(self, batch, sde_step=50):
        z, b, x, abid, edge_mask, bond_index = batch["atype"], batch["btype"], batch["x0"], batch["abid"], batch["edge_mask"], batch["bond_index"]
        post_cross_condition = None
        pvb_state = None
        N = x.shape[0]

        # encode
        block_condition = None
        if self.fusion_mode in {"off", "anew_block_pvb_posterior", "pvb_shared_hblock"}:
            # Preserve the original PVB posterior sampling path for both the
            # baseline and corrected mode.
            if self.fusion_mode == "pvb_shared_hblock":
                x_rep, _, pvb_state = self._encode_pvb_path(
                    z, b, x, x, abid, torch.ones_like(z).bool(), edge_mask,
                    bond_index, return_state=True,
                )
            else:
                x_rep, _ = self._encode_pvb_path(
                    z, b, x, x, abid, torch.ones_like(z).bool(), edge_mask,
                    bond_index,
                )
            if self.fusion_mode == "anew_block_pvb_posterior":
                block_output = self._encode_anew_conditioning(x, batch)
                block_condition = self._project_block_condition(block_output)
            elif self.fusion_mode == "pvb_shared_hblock":
                post_cross_condition, _ = self._project_shared_hblock_condition(
                    pvb_state["h_atom"], batch
                )
        else:
            # Legacy Phase 9 semantics use Anew's posterior here.
            x_rep, _, block_output = self._encode_anew_block(
                x, batch, torch.ones_like(z).bool()
            )
            block_condition = self._project_block_condition(block_output)

        xt = x_rep.clone()
        sde_step = np.linspace(0, 1. - 1. / sde_step, sde_step)
        dt = sde_step[1] - sde_step[0]

        t_all_one = torch.ones(N, 1, dtype=x.dtype, device=x.device)

        for t in sde_step:
            t_diff = t_all_one * t
            # construct edges
            d_edge_index, d_edge_weight_t, d_edge_vec_t, bond_type = construct_edges(
                Z=z, X=xt, bid=abid, mask=edge_mask, bond_index=bond_index,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                cutoff_H=self.cutoff_H,
                k_neighbors=self.k_neighbors
            )
            d_edge_atm_0 = x_rep[d_edge_index.transpose(0, 1)]      # (Ef, 2, 3)
            d_edge_vec_0 = d_edge_atm_0[:, 0] - d_edge_atm_0[:, 1]  # (Ef, 3)
            d_edge_weight_0 = torch.norm(d_edge_vec_0, dim=-1)      # (Ef,)
            if self.using_ode:
                vel_pred = self.decode(z, b, xt, t_diff, abid, d_edge_index,
                                       d_edge_weight_0, d_edge_vec_0,
                                       d_edge_weight_t, d_edge_vec_t,
                                       bond_type, block_condition=block_condition,
                                       post_cross_condition=post_cross_condition)
                xt = xt + vel_pred * dt
            else:
                vel_pred, drf_pred = self.decode(z, b, xt, t_diff, abid, d_edge_index,
                                                 d_edge_weight_0, d_edge_vec_0,
                                                 d_edge_weight_t, d_edge_vec_t,
                                                 bond_type, block_condition=block_condition,
                                                 post_cross_condition=post_cross_condition)
                # drift = INTERP_MATCHER.drift(vel_pred, den_pred, t, self.sigma)
                drift = drf_pred
                diffusion = INTERP_MATCHER.diffusion(self.sigma)
                xt = xt + drift * dt + diffusion * np.sqrt(dt) * torch.randn_like(xt) if t != sde_step[-1] \
                    else xt + drift * dt
        return xt

    def realization(self, batch, sde_step=50):
        z, b, x0, abid, edge_mask, bond_index = batch["atype"], batch["btype"], batch["x0"], batch["abid"], batch["edge_mask"], batch["bond_index"]
        N = x0.shape[0]

        # given x0
        xt = x0.clone()
        block_condition = None
        post_cross_condition = None
        if self.fusion_mode == "pvb_shared_hblock":
            # The shared Phase 11 path reuses PVB's scalar encoder state for
            # decoder-only conditioning; no Anew encoder or posterior is run.
            pvb_diagnostics = self._pvb_encoder_diagnostics(
                z,
                b,
                x0,
                abid,
                torch.ones_like(z).bool(),
                edge_mask,
                bond_index,
            )
            post_cross_condition, _ = self._project_shared_hblock_condition(
                pvb_diagnostics["H_pvb"], batch
            )
        elif self.fusion_mode in {"anew_block", "anew_block_pvb_posterior"}:
            block_output = self.anew_block_encoder(
                x_atom=x0,
                atom_type=batch["atype"],
                block_type=batch["block_type"],
                atom_block_id=batch["atom_block_id"],
                block_batch=batch["block_batch"],
                block_lengths=batch["block_lengths"],
                bond_index=batch.get("bond_index"),
                bond_type=batch.get("bond_type"),
            )
            block_condition = self._project_block_condition(block_output)
        trajs = [x0.detach().clone()]
        sde_step = np.linspace(0, 1. - 1. / sde_step, sde_step)
        dt = sde_step[1] - sde_step[0]

        t_all_one = torch.ones(N, 1, dtype=x0.dtype, device=x0.device)

        for t in sde_step:
            t_diff = t_all_one * t
            # construct edges
            d_edge_index, d_edge_weight_t, d_edge_vec_t, bond_type = construct_edges(
                Z=z, X=xt, bid=abid, mask=edge_mask, bond_index=bond_index,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                cutoff_H=self.cutoff_H,
                k_neighbors=self.k_neighbors
            )
            d_edge_atm_0 = x0[d_edge_index.transpose(0, 1)]      # (Ef, 2, 3)
            d_edge_vec_0 = d_edge_atm_0[:, 0] - d_edge_atm_0[:, 1]  # (Ef, 3)
            d_edge_weight_0 = torch.norm(d_edge_vec_0, dim=-1)      # (Ef,)
            if self.using_ode:
                vel_pred = self.decode(z, b, xt, t_diff, abid, d_edge_index,
                                       d_edge_weight_0, d_edge_vec_0,
                                       d_edge_weight_t, d_edge_vec_t,
                                       bond_type, block_condition=block_condition,
                                       post_cross_condition=post_cross_condition)
                xt = xt + vel_pred * dt
            else:
                vel_pred, drf_pred = self.decode(z, b, xt, t_diff, abid, d_edge_index,
                                                 d_edge_weight_0, d_edge_vec_0,
                                                 d_edge_weight_t, d_edge_vec_t,
                                                 bond_type, block_condition=block_condition,
                                                 post_cross_condition=post_cross_condition)
                # drift = INTERP_MATCHER.drift(vel_pred, den_pred, t, self.sigma)
                drift = drf_pred
                diffusion = INTERP_MATCHER.diffusion(self.sigma)
                xt = xt + drift * dt + diffusion * np.sqrt(dt) * torch.randn_like(xt) if t != sde_step[-1] \
                    else xt + drift * dt
            trajs.append(xt.detach().clone())
        return trajs, sde_step


class dyAdjMatch(nn.Module):
    def __init__(self, ref_model, reward_weight=100, rl_sde_step=50):
        super().__init__()

        self.opt: dyVAE = deepcopy(ref_model)
        self.ref: dyVAE = deepcopy(ref_model)
        for param in self.ref.parameters():
            param.requires_grad = False
        
        self.cutoff_lower = self.ref.cutoff_lower
        self.cutoff_upper = self.ref.cutoff_upper
        self.cutoff_H = self.ref.cutoff_H
        self.k_neighbors = self.ref.k_neighbors
        self.sigma = self.ref.sigma
        self.additional_noise_scale = self.ref.additional_noise_scale
        self.reward_weight = reward_weight
        self.rl_sde_step = rl_sde_step
    
    def _train(self, batch):
        # obtain x0
        z, b, x, x_ref, abid, mask, edge_mask, bond_index = batch["atype"], batch["btype"], batch["x0"], batch["x_ref"], batch["abid"], batch["mask"], batch["edge_mask"], batch["bond_index"]
        N = x.shape[0]

        # perform random SE(3) transformation
        if self.training:
            for i in range(abid[-1] + 1):
                ligand_index = torch.where(
                    torch.logical_and((edge_mask == 1), (abid == i))
                )[0]
                R = torch.from_numpy(random_rotation_matrix(angle_limit=np.pi / 3)) # np.pi / 6
                t = torch.randn(3) / np.sqrt(3)    # 1.0 Angstrom
                R, t = R.float().to(x.device), t.float().to(x.device)
                xc = x[ligand_index].mean(dim=0)
                x[ligand_index] = (x[ligand_index] - xc) @ R + xc + t

        with torch.no_grad():
            # encode
            # construct edges
            e_edge_index, e_edge_weight, e_edge_vec, bond_type = construct_edges(
                Z=z, X=x, bid=abid, mask=edge_mask, bond_index=bond_index,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                cutoff_H=self.cutoff_H,
                k_neighbors=self.k_neighbors
            )
            x_rep, _ = self.opt.encode(z, b, x, x, abid, e_edge_index, e_edge_weight, e_edge_vec, bond_type, torch.ones_like(z).bool())
            batch["x0"] = x_rep.detach().clone()
            # generate a realization from p_{\theta}
            trajs, dts = self.opt.realization(batch, sde_step=self.rl_sde_step)

        with torch.enable_grad():
            # compute lean adjoint state
            x1 = trajs[-1]
            x1.requires_grad = True
            rewards = []
            for i in range(abid[-1] + 1):
                rewards.append(compute_crmsd_torch(x1[abid == i], x_ref[abid == i], aligned=False))
            rewards = torch.stack(rewards)
            a = compute_grad(rewards, x1, create_graph=False).detach()

            # decode: bridge matching
            t_all_one = torch.ones(N, 1, dtype=x.dtype, device=x.device)
            # uniformly sample time step
            t_idx = np.random.choice(range(len(dts)))
            t = dts[t_idx]
            # solve backward SDE
            dts = np.append(dts, 1.0)   # [0.0, 0.02, ..., 1.0]
            for i in reversed(range(t_idx + 1, len(dts))):
                s, ds = dts[i], dts[i] - dts[i - 1]
                t_diff = t_all_one * s
                xs = trajs[i]
                xs.requires_grad = True
                # construct edges
                d_edge_index, d_edge_weight_s, d_edge_vec_s, bond_type = construct_edges(
                    Z=z, X=xs, bid=abid, mask=edge_mask, bond_index=bond_index,
                    cutoff_lower=self.cutoff_lower,
                    cutoff_upper=self.cutoff_upper,
                    cutoff_H=self.cutoff_H,
                    k_neighbors=self.k_neighbors
                )
                d_edge_atm_0 = x_rep[d_edge_index.transpose(0, 1)]      # (Ef, 2, 3)
                d_edge_vec_0 = d_edge_atm_0[:, 0] - d_edge_atm_0[:, 1]  # (Ef, 3)
                d_edge_weight_0 = torch.norm(d_edge_vec_0, dim=-1)      # (Ef,)
                # compute drift term of p_{ref}
                _, drf_pred = self.ref.decode(z, b, xs, t_diff, abid, d_edge_index,
                                            d_edge_weight_0, d_edge_vec_0,
                                            d_edge_weight_s, d_edge_vec_s,
                                            bond_type)
                # compute vector-Jaccobian product (vjp)
                vjp = compute_grad(drf_pred, xs, grad_outputs=a, create_graph=False).detach() # (N, 3)
                a = a + ds * vjp
        # get the lean adjoint state $a$ w.r.t. $t$
        t_diff = t_all_one * t
        xt = trajs[t_idx]
        # construct edges
        d_edge_index, d_edge_weight_t, d_edge_vec_t, bond_type = construct_edges(
            Z=z, X=xt, bid=abid, mask=edge_mask, bond_index=bond_index,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            cutoff_H=self.cutoff_H,
            k_neighbors=self.k_neighbors
        )
        d_edge_atm_0 = x_rep[d_edge_index.transpose(0, 1)]      # (Ef, 2, 3)
        d_edge_vec_0 = d_edge_atm_0[:, 0] - d_edge_atm_0[:, 1]  # (Ef, 3)
        d_edge_weight_0 = torch.norm(d_edge_vec_0, dim=-1)      # (Ef,)
        # compute drift term of p_{ref}
        _, drf_pred_ref = self.ref.decode(z, b, xt, t_diff, abid, d_edge_index,
                                          d_edge_weight_0, d_edge_vec_0,
                                          d_edge_weight_t, d_edge_vec_t,
                                          bond_type)
        # compute drift term of p_{opt}
        _, drf_pred_opt = self.opt.decode(z, b, xt, t_diff, abid, d_edge_index,
                                          d_edge_weight_0, d_edge_vec_0,
                                          d_edge_weight_t, d_edge_vec_t,
                                          bond_type)
        # diffusion = INTERP_MATCHER.diffusion(self.sigma)
        adj_match_loss = F.mse_loss(drf_pred_opt[mask], drf_pred_ref[mask] - self.reward_weight * a[mask])

        loss = 1.0 * adj_match_loss

        return loss, (adj_match_loss,)
    
    def inference(self, batch, sde_step=50):
        return self.opt.inference(batch, sde_step=sde_step)


class GeoDPO(nn.Module):
    def __init__(self, ref_model, beta=2.0, rl_sde_step=10):
        super().__init__()
        
        self.opt: dyVAE = deepcopy(ref_model)
        self.ref: dyVAE = deepcopy(ref_model)
        for param in self.ref.parameters():
            param.requires_grad = False
        
        self.cutoff_lower = self.ref.cutoff_lower
        self.cutoff_upper = self.ref.cutoff_upper
        self.cutoff_H = self.ref.cutoff_H
        self.k_neighbors = self.ref.k_neighbors
        self.sigma = self.ref.sigma
        self.additional_noise_scale = self.ref.additional_noise_scale
        self.beta = beta
        self.rl_sde_step = rl_sde_step

    def _train(self, batch):
        # generate two target states
        with torch.no_grad():
            x1w = self.opt.inference(batch, sde_step=self.rl_sde_step)
            x1l = self.opt.inference(batch, sde_step=self.rl_sde_step)
        # compare with reference conformations
        x_ref = batch["x_ref"]
        # TODO: fix bug: different confs in one batch
        rmsd_w = compute_crmsd(x1w, x_ref)
        rmsd_l = compute_crmsd(x1l, x_ref)
        if rmsd_w > rmsd_l:
            x1w, x1l = x1l, x1w

        x0, b0, atype, btype, abid, mask, edge_mask, bond_index = batch["x0"], batch["b0"], batch["atype"], batch["btype"], batch["abid"], batch["mask"], batch["edge_mask"], batch["bond_index"]
        # specialize if CUDA out of memory
        x0, x1w, x1l, atype, btype, abid, edge_mask = x0[mask], x1w[mask], x1l[mask], atype[mask], btype[mask], abid[mask], edge_mask[mask]
        max_indices_list = torch.where(mask == 1)[0].tolist()
        # re-index bond index
        index_mapping = {x: max_indices_list.index(x) for x in max_indices_list}
        bond_index_slice = []
        src, dst = bond_index.tolist()
        for begin, end in zip(src, dst):
            begin = index_mapping.get(begin, -1)
            end = index_mapping.get(end, -1)
            if begin == -1 or end == -1:
                continue
            bond_index_slice.append([begin, end])
        bond_index = torch.tensor(bond_index_slice, dtype=torch.long, device=x0.device).T
        if bond_index.numel() == 0:
            # torch.distributed.barrier()
            pdb.set_trace()
        mask = torch.ones_like(atype).bool()
        # each heavy atom as mu
        b0 = x0.clone()
        
        N = x0.shape[0]
        with torch.no_grad():
            # encode
            # construct edges
            e_edge_index, e_edge_weight, e_edge_vec, e_bond_type = construct_edges(
                Z=atype, X=x0, bid=abid, mask=edge_mask, bond_index=bond_index,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                cutoff_H=self.cutoff_H,
                k_neighbors=self.k_neighbors
            )
            x_rep, _ = self.opt.encode(atype, btype, x0, b0, abid, e_edge_index, e_edge_weight, e_edge_vec, e_bond_type, mask)
        
        # decode: bridge matching
        t_all_one = torch.ones(N, 1, dtype=x0.dtype, device=x0.device)
        # uniformly sample time step
        t = np.random.rand()  # use the same time step for all mini-batches
        t_diff = t_all_one * t

        # sample interpolants
        xtw, vel_gt_w, den_gt_w = INTERP_MATCHER.sample_conditional_flow(x_rep, x1w, t_diff, self.sigma)
        xtl, vel_gt_l, den_gt_l = INTERP_MATCHER.sample_conditional_flow(x_rep, x1l, t_diff, self.sigma)
        drf_gt_w = INTERP_MATCHER.drift(vel_gt_w, den_gt_w, t_diff, self.sigma)
        drf_gt_l = INTERP_MATCHER.drift(vel_gt_l, den_gt_l, t_diff, self.sigma)

        # construct edges for positive candidate
        d_edge_index_w, d_edge_weight_w, d_edge_vec_w, bond_type_w = construct_edges(
            Z=atype, X=xtw, bid=abid, mask=edge_mask, bond_index=bond_index,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            cutoff_H=self.cutoff_H,
            k_neighbors=self.k_neighbors
        )
        d_edge_atm_w_0 = x_rep[d_edge_index_w.transpose(0, 1)]        # (Ef, 2, 3)
        d_edge_vec_w_0 = d_edge_atm_w_0[:, 0] - d_edge_atm_w_0[:, 1]  # (Ef, 3)
        d_edge_weight_w_0 = torch.norm(d_edge_vec_w_0, dim=-1)        # (Ef,)

        # construct edges for negative candidate
        d_edge_index_l, d_edge_weight_l, d_edge_vec_l, bond_type_l = construct_edges(
            Z=atype, X=xtl, bid=abid, mask=edge_mask, bond_index=bond_index,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            cutoff_H=self.cutoff_H,
            k_neighbors=self.k_neighbors
        )
        d_edge_atm_l_0 = x_rep[d_edge_index_l.transpose(0, 1)]        # (Ef, 2, 3)
        d_edge_vec_l_0 = d_edge_atm_l_0[:, 0] - d_edge_atm_l_0[:, 1]  # (Ef, 3)
        d_edge_weight_l_0 = torch.norm(d_edge_vec_l_0, dim=-1)        # (Ef,)

        # reference model
        vf_ref_w, drf_ref_w = self.ref.decode(atype, btype, xtw, t_diff, abid, d_edge_index_w,
                                              d_edge_weight_w_0, d_edge_vec_w_0,
                                              d_edge_weight_w, d_edge_vec_w,
                                              bond_type_w)
        vf_ref_l, drf_ref_l = self.ref.decode(atype, btype, xtl, t_diff, abid, d_edge_index_l,
                                              d_edge_weight_l_0, d_edge_vec_l_0,
                                              d_edge_weight_l, d_edge_vec_l,
                                              bond_type_l)

        # model to be optimized
        vf_opt_w, drf_opt_w = self.opt.decode(atype, btype, xtw, t_diff, abid, d_edge_index_w,
                                              d_edge_weight_w_0, d_edge_vec_w_0,
                                              d_edge_weight_w, d_edge_vec_w,
                                              bond_type_w)
        vf_opt_l, drf_opt_l = self.opt.decode(atype, btype, xtl, t_diff, abid, d_edge_index_l,
                                              d_edge_weight_l_0, d_edge_vec_l_0,
                                              d_edge_weight_l, d_edge_vec_l,
                                              bond_type_l)

        dpo_loss = - torch.log(
            F.sigmoid(
                - self.beta * (
                    F.mse_loss(drf_opt_w[mask], drf_gt_w[mask], reduction='none').sum(dim=1) - 
                    F.mse_loss(drf_ref_w[mask], drf_gt_w[mask], reduction='none').sum(dim=1) -
                    F.mse_loss(drf_opt_l[mask], drf_gt_l[mask], reduction='none').sum(dim=1) +
                    F.mse_loss(drf_ref_l[mask], drf_gt_l[mask], reduction='none').sum(dim=1)
                )
            )
        ).mean()
        
        loss = 1.0 * dpo_loss

        return loss, (dpo_loss,)
    
    def inference(self, batch, sde_step=50):
        return self.opt.inference(batch, sde_step=sde_step)
