#!/usr/bin/env python
"""Which ABSTRACT STATISTICAL FAILURE MODE does each documented trap instantiate?

Built 2026-09-21 from the full enumeration of all 25 families' traps. See docs/CROSS_DOMAIN_TASK_DESIGN.md.

WHY THIS FILE EXISTS. Every family names its traps in its own vocabulary, so nothing in this programme has ever
been able to ask "does the agent handle THIS KIND OF ERROR", only "does it handle THIS FAMILY". That made
cross-domain transfer untestable by construction: a transfer test always moved to a family whose traps shared no
content with the source, so a null was guaranteed whatever the agent had learned.

Labelling is the cheap half of the fix. With it, every result already on disk becomes re-analysable BY MODE at
zero measurement cost, and `trap_sweep.py` can report per-mode instead of per-family.

The mapping is a scientific claim, not bookkeeping: two traps share a mode when the SAME statistical mistake is
being made about different subject matter. Where a trap is ambiguous it is left UNMAPPED rather than forced --
an inflated grid would manufacture transfer that is not there.
"""

MODES = {
    "M1_censoring":     "a value observed only partially is treated as complete",
    "M2_heterogeneity": "units that are not exchangeable are pooled",
    "M3_functional":    "a nonlinear process is forced into a linear fit",
    "M4_measurement":   "the measuring process (units, calibration, drift, ascertainment) is not modelled",
    "M5_selection":     "conditioning on a post-exposure variable or on a selected sample",
    "M6_identification":"an available exogenous source of variation is ignored",
    "M7_entity":        "distinct entities are counted or matched wrongly",
}

# THREE ACCEPTED VARIANTS REMOVED 2026-09-21, verified from the registry rather than by hand: a trap whose
# `Trap.step` is None is an alternative analysis the grader TOLERATES, not a mistake -- counting it inflates the
# per-mode family counts. Removed: blq_discarded (pk_oral, was M1), logistic_referral_model (hc_dx, was M3),
# hospital_indicators (surv_cox, was M5). `variants_in_mapping()` below re-runs this check against the registry.
TRAP_MODE = {
    # M1 -- partial observation treated as complete
    "no_truncation_correction": "M1_censoring", "blq_as_zero": "M1_censoring",
    "censored_as_deaths": "M1_censoring",
    "deaths_only": "M1_censoring",
    # M2 -- pooling units that differ
    "ignore_hospital": "M2_heterogeneity", "crude_no_casemix": "M2_heterogeneity",
    "m0_equal_catchability": "M2_heterogeneity", "ignore_sex": "M2_heterogeneity",
    "additive_ols": "M2_heterogeneity", "no_pcs": "M2_heterogeneity",
    "keep_related": "M2_heterogeneity", "no_pcs_keep_related": "M2_heterogeneity",
    "no_batch_adjustment": "M2_heterogeneity", "all_cells_no_state": "M2_heterogeneity",
    "pooled_gc": "M2_heterogeneity", "pooled_distance": "M2_heterogeneity",
    "no_weekday_effects": "M2_heterogeneity", "ignore_trap_response": "M2_heterogeneity",
    # M3 -- wrong functional form
    "lineweaver_burk": "M3_functional", "linear_whole_curve": "M3_functional",
    "single_exponential": "M3_functional", "global_linear": "M3_functional",
    "nonlinear_unweighted": "M3_functional", "undamped_sinusoid": "M3_functional",
    "unweighted_linear": "M3_functional",
    # M4 -- the measuring process
    "celsius_temperatures": "M4_measurement", "calibration_offset_ignored": "M4_measurement",
    "ignore_testing_change": "M4_measurement", "no_detrending": "M4_measurement",
    "no_unit_conversion": "M4_measurement", "units_not_harmonised": "M4_measurement",
    "raw_counts": "M4_measurement", "no_library_offset": "M4_measurement",
    "no_ambient_correction": "M4_measurement", "no_limb_darkening": "M4_measurement",
    "no_outlier_removal": "M4_measurement",
    # M5 -- selection and post-exposure conditioning
    "post_treatment_control": "M5_selection", "ever_treated_from_diagnosis": "M5_selection",
    "referred_only": "M5_selection", "unreferred_as_negative": "M5_selection",
    "referral_by_test_only": "M5_selection", "in_hospital_deaths_only": "M5_selection",
    "hospital_in_casemix_model": "M5_selection", "panel_maf_filter": "M5_selection", "keep_sv_stripe": "M5_selection",
    # M6 -- identification ignored.  ONE trap in ONE family. Two mappings were REMOVED 2026-09-21 after an
    # adversarial review, and removing them is what makes the honest statement possible:
    #   `reduced_form`  USES the instrument (Y on Z) and fails to scale by the first stage -- a scaling/estimand
    #                   error, the same class as `iv_without_covariates`, which was already unmapped.
    #   `univariable` (cis_mvmr) USES all the instruments; what it omits is the correlated co-exposure, i.e.
    #                   omitted-variable bias -- the class `no_adjustment`/`confounder_omitted` is unmapped for.
    # Consequence, stated plainly: **M6 spans ONE family**, so the "IV in economics vs Mendelian randomisation in
    # genomics" cell in the design does not exist in this registry, and item 389 was pulled before it ran.
    "ols_ignores_instrument": "M6_identification",
    # M7 -- entity resolution
    "tag_loss_unmerged": "M7_entity", "distinct_count": "M7_entity",
    "exact_string_match": "M7_entity", "substring_overmatch": "M7_entity",
    "no_clumping": "M7_entity", "keep_unmapped": "M7_entity",
}

# Deliberately UNMAPPED, with the reason, so the grid is not inflated:
#   no_adjustment, confounder_omitted   -- plain omitted-variable bias; its own mode, and it appears in only the
#                                          causal families, so mapping it would add no cross-domain reach
#   treated_before_after, post_period_gap, difference_in_means -- "no counterfactual"; arguably a mode (M8) but it
#                                          lives only in causal_did/causal_rdd, so it cannot support a transfer test
#   ate_not_att, iv_without_covariates  -- right family, wrong estimand: an estimand error, not an analysis error
#   half_period, phase_half, ls_peak, first_acf_peak, acf_peak_series, swapped_eclipses -- periodicity aliasing,
#                                          astronomy-only in this registry
#   one_compartment, two_compartment_forced, half_life_from_early_decline -- model-order selection, pk-only
#   iv_without_covariates, screening_betas, ld_ignored, no_residual_pruning, threshold_5 -- MR pipeline specifics
#   auc_of_binary_result, raw_smr_spread, any_recorded_death, raw_score_*, include_synonymous, no_maf_filter,
#   no_pvalue_filter, lead_by_effect, raw_or_beta, beta_density_weights, no_lowmap_mask, failed_runs_included,
#   no_supersampling, fft_interpolated, damped_frequency_as_f0 -- either family-specific or a threshold choice


def mode_of(trap):
    return TRAP_MODE.get(trap)


def coverage():
    """Which modes span enough DOMAINS to support a held-out transfer test?"""
    import collections, warnings, numpy as np
    warnings.filterwarnings("ignore")
    from sage.registry import families
    by_mode = collections.defaultdict(set)
    F = families()
    for name in sorted(F):
        try:
            f = F[name]; rng = np.random.default_rng(0)
            traps = f.traps(f.draw(rng, f.knobs(rng)).data)
        except Exception:
            continue
        for t in traps:
            m = mode_of(t)
            if m:
                by_mode[m].add(name)
    return by_mode


if __name__ == "__main__":
    print(f"{len(TRAP_MODE)} traps mapped to {len(MODES)} modes\n")
    for m, fams in sorted(coverage().items(), key=lambda kv: -len(kv[1])):
        flag = "  <- supports a held-out transfer test" if len(fams) >= 4 else ""
        print(f"{m:<20} {len(fams)} famil(y/ies): {', '.join(sorted(fams))}{flag}")
        print(f"{'':<20} {MODES[m]}")


def variants_in_mapping():
    """Which mapped traps are ACCEPTED VARIANTS (Trap.step is None), not errors?

    Recorded as owed on 2026-09-21: the mapping was curated by hand, and a trap whose `step` is None is an
    accepted alternative analysis the grader tolerates (e.g. `hospital_indicators`, `blq_discarded`,
    `logistic_referral_model`), not a mistake. Counting them as traps inflates the per-mode family counts that
    section 7 of docs/CROSS_DOMAIN_TASK_DESIGN.md reports. This reads the registry rather than trusting the dict.
    """
    import warnings, collections
    warnings.filterwarnings("ignore")
    import numpy as np
    from sage.registry import families
    bad = collections.defaultdict(list)
    F = families()
    for name in sorted(F):
        try:
            f = F[name]; rng = np.random.default_rng(0)
            traps = f.traps(f.draw(rng, f.knobs(rng)).data)
        except Exception:
            continue
        for t, spec in traps.items():
            if t in TRAP_MODE and getattr(spec, "step", "missing") is None:
                bad[TRAP_MODE[t]].append(f"{t} ({name})")
    return bad
