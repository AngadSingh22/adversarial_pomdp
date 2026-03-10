# Mathematical Framework Audit

Audit target: `formulation/paper/main.tex`

Scope: line-by-line review of the formal definitions, theorem statements, proofs, and theorem-dependent empirical interpretations.

## Critical Errors

### 1. The paper proves an undiscounted objective, but the implementation trains a discounted PPO objective

Relevant lines:

- `main.tex:105-128`
- `main.tex:822-827`

Issue:

- The lemma at `main.tex:119-128` claims that the step-penalty reward yields
  \[
  J(\pi)=-\mathbb{E}_\pi[\tau].
  \]
  That identity is true only for the undiscounted return.
- The manuscript explicitly includes a discount factor in the POMDP tuple at `main.tex:108` and later reports PPO with `\gamma=0.99` at `main.tex:826`.
- Therefore the attacker actually optimizes
  \[
  J_\gamma(\pi)=\mathbb{E}_\pi\!\left[\sum_{t=0}^{\tau-1}\gamma^t(-1)\right]
  =-\mathbb{E}_\pi\!\left[\frac{1-\gamma^\tau}{1-\gamma}\right],
  \]
  not `-\mathbb{E}_\pi[\tau]`.

Why this is serious:

- The objective used in the proofs is not the objective optimized by the reported algorithm.
- The approximate best-response certificates at `main.tex:228-254` are written for expected episode length. They do not automatically apply to a policy trained against the discounted surrogate.
- Because \(\tau \mapsto (1-\gamma^\tau)/(1-\gamma)\) is nonlinear, policy rankings under discounted return and expected episode length can differ.

Concrete counterexample:

- Let \(\gamma=1/2\).
- Policy A: \(\tau=2\) almost surely. Then \(\mathbb{E}[\tau]=2\) and \(\mathbb{E}[\sum_{t=0}^{\tau-1}\gamma^t]=1.5\).
- Policy B: \(\tau=1\) with probability \(1/2\) and \(\tau=4\) with probability \(1/2\). Then \(\mathbb{E}[\tau]=2.5\) but
  \[
  \mathbb{E}\!\left[\sum_{t=0}^{\tau-1}\gamma^t\right]
  =\tfrac12\cdot 1+\tfrac12\cdot(1+1/2+1/4+1/8)=1.4375.
  \]
- Thus discounted return prefers the policy with worse expected shots-to-win.

Corrected LaTeX derivation:

```latex
\begin{lemma}[Discounted step penalty]
Suppose
\[
r_t=-1 \quad \text{for } t<\tau, \qquad r_\tau=0.
\]
Then for any policy $\pi$ and any discount factor $\gamma\in(0,1)$,
\[
J_\gamma(\pi)
:=
\mathbb{E}_\pi\!\left[\sum_{t=0}^{\tau-1}\gamma^t r_t\right]
=
-\mathbb{E}_\pi\!\left[\sum_{t=0}^{\tau-1}\gamma^t\right]
=
-\mathbb{E}_\pi\!\left[\frac{1-\gamma^\tau}{1-\gamma}\right].
\]
In particular, $J_\gamma(\pi)=-\mathbb{E}_\pi[\tau]$ only when $\gamma=1$.
\end{lemma}
```

Required fix:

- Either set the training discount to \(\gamma=1\) and keep the current theorem package

### 2. The Stage-2 table is not algebraically consistent with the paper's own diagnostic definitions

Relevant lines:

- `main.tex:232-252`
- `main.tex:408-418`
- `main.tex:534-536`

Issue:

- The notation table defines
  \[
  \textsf{defender\_adversarial}_k
  =

  \mathbb{E}_{\rho_k}[\tau(\pi_{k-1})]
  -

  \mathbb{E}_{\rho_U}[\tau(\pi_{k-1})].
  \]
- But the displayed Stage-2 numbers do not satisfy that identity.

Example:

- Seed 42, generation 1 in Table 2 reports `UNIFORM = 90.3`, `D_k = 90.5`, and `defender_adversarial = -2.15`.
- If `UNIFORM` and `D_k` are the two quantities entering the diagnostic definition, then the diagnostic should be \(90.5-90.3=0.2\), not \(-2.15\).
- The same mismatch appears in other rows.

Why this is serious:

- The table is presented as the main empirical bridge to Theorem 2 at `main.tex:399-426`.
- As written, a reader cannot reconstruct the diagnostics from the reported numbers.
- This prevents verification of the claimed attacker/defender certificate interpretation.

Minimum correction:

- The table must expose the exact quantities used by the definitions:

```latex
\begin{tabular}{ccccccccc}
\toprule
Seed & k &
\mathbb{E}_{\rho_U}[\tau(\pi_{k-1})] &
\mathbb{E}_{\rho_k}[\tau(\pi_{k-1})] &
\mathbb{E}_{\rho_U}[\tau(\pi_k)] &
\mathbb{E}_{\rho_k}[\tau(\pi_k)] &
\textsf{defender\_adversarial}_k &
\textsf{attacker\_adaptation}_k &
\textsf{uniform\_drift}_k \\
\midrule
\cdots
\end{tabular}
```

with

```latex
\[
\textsf{defender\_adversarial}_k
=
\mathbb{E}_{\rho_k}[\tau(\pi_{k-1})]
-\mathbb{E}_{\rho_U}[\tau(\pi_{k-1})],
\]
\[
\textsf{attacker\_adaptation}_k
=
\mathbb{E}_{\rho_k}[\tau(\pi_k)]
-\mathbb{E}_{\rho_k}[\tau(\pi_{k-1})],
\]
\[
\textsf{uniform\_drift}_k
=
\mathbb{E}_{\rho_U}[\tau(\pi_k)]
-\mathbb{E}_{\rho_U}[\tau(\pi_{k-1})].
\]
```

### 3. The empirical claims invoke certificate theorems without reporting the quantities needed to evaluate those certificates

Relevant lines:

- `main.tex:228-306`
- `main.tex:399-426`

Issue:

- The approximate best-response theorem requires \(\varepsilon_D\), \(\varepsilon_A\), and \(\lambda\).
- The finite-sample sign theorem requires \(n_D\), \(n_U\), and analogous sample counts for the weighted attacker residual.
- None of these quantities are reported in the theorem-dependent tables or discussion.

Why this is serious:

- Statements such as "entirely consistent with our approximate certificate theory" and "the finite-sample theorem explains when the empirical signs of these quantities are trustworthy" are not presently checkable from the manuscript.
- A negative `defender_adversarial` value only implies a defender failure if the reader knows the size of \(\varepsilon_D\) or at least a meaningful upper bound.
- Likewise, sign-certification cannot be claimed without the confidence radius.

What must be added:

- The paper should report \(\lambda\), the optimization tolerances or computable proxies for \(\varepsilon_A,\varepsilon_D\), and the evaluation sample counts.
- Each theorem-dependent table should include the implied bound alongside the observed diagnostic.

### 4. The finite-sample theorem states an "analogous" weighted-residual bound, but the bound is never actually stated

Relevant lines:

- `main.tex:271-305`
- `main.tex:741-747`

Issue:

- The theorem at `main.tex:302-305` says an analogous bound holds for the weighted attacker residual.
- The proof then refers to "the stated weighted confidence radius" at `main.tex:747`, but no explicit weighted confidence radius appears in the theorem statement.

Why this matters:

- This is a proof-statement mismatch.
- A reader cannot apply the theorem to the attacker residual without reconstructing the omitted formula.

Corrected LaTeX derivation:

```latex
\[
\widehat{R}_k
=
\lambda(\overline{X}_1-\overline{X}_2)
+(1-\lambda)(\overline{Y}_1-\overline{Y}_2),
\]
\[
R_k
=
\lambda\!\left(\mathbb{E}[\overline{X}_1]-\mathbb{E}[\overline{X}_2]\right)
+(1-\lambda)\!\left(\mathbb{E}[\overline{Y}_1]-\mathbb{E}[\overline{Y}_2]\right).
\]
If each empirical mean is computed from independent samples in $[0,T_{\max}]$, then for any
$\delta\in(0,1)$, with probability at least $1-\delta$,
\[
|\widehat{R}_k-R_k|
\le
\lambda T_{\max}
\left(
\sqrt{\frac{\log(8/\delta)}{2n_{X_1}}}
+
\sqrt{\frac{\log(8/\delta)}{2n_{X_2}}}
\right)
+(1-\lambda) T_{\max}
\left(
\sqrt{\frac{\log(8/\delta)}{2n_{Y_1}}}
+
\sqrt{\frac{\log(8/\delta)}{2n_{Y_2}}}
\right).
\]
```

## Suggested Enhancements

### 1. Tighten notation to remove overloaded symbols

Relevant lines:

- `main.tex:105-145`
- `main.tex:182-224`

Problems:

- `\mathcal{Z}` is used in the Battleship POMDP tuple at `main.tex:108`, but later `z\in\mathcal{Z}` is the latent state itself.
- `\mu` denotes an attacker mixture in Theorem 1 at `main.tex:185-191`, then a defender-side mixture at `main.tex:220-224`.

Recommendation:

- Use a separate symbol such as \(\Theta\) for the latent parameter space.
- Use \(\Omega\) for the observation kernel/set if standard POMDP notation is intended.
- Use distinct symbols such as \(\sigma\) for attacker mixtures and \(\nu\) for defender mixtures.

### 2. Make the legal-action dependence explicit in the policy definition

Relevant lines:

- `main.tex:165-170`

Issue:

- A map \(\pi:\mathcal{H}\to\mathcal{A}\) is too coarse because the feasible action set depends on history.

Recommended correction:

```latex
\begin{definition}[Deterministic history-dependent attacker policy]
A deterministic history-dependent attacker policy is a map
\[
\pi:\mathcal{H}\to\mathcal{C}
\]
such that for every feasible history $h\in\mathcal{H}$,
\[
\pi(h)\in \mathcal{A}(h).
\]
\end{definition}
```

### 3. Define the tail functionals formally

Relevant lines:

- `main.tex:149-160`

Issue:

- `p95` and `\mathrm{CVaR}_{0.10}` are used without a formal definition.
- For a loss variable, readers need to know whether `\mathrm{CVaR}_{0.10}` refers to the upper 10% tail or the lower 10% tail.

Recommendation:

- Add a one-line formal definition, for example
  \[
  \mathrm{CVaR}^{\mathrm{upper}}_{0.10}(X)
  =

  \frac{1}{0.10}\int_{0.90}^{1}F_X^{-1}(u)\,du.
  \]

### 4. The Pareto corollary should either be proved or cited

Relevant lines:

- `main.tex:330-335`

Issue:

- The scalarization claim is standard multiobjective optimization, but in a math/CS journal it should not be left as an unproved folklore statement.

Recommendation:

- Add a short proof that any exact minimizer of a strictly positive weighted sum is weakly Pareto-optimal, and "supported" because the minimizing point is exposed by the supporting hyperplane with normal \((\lambda,1-\lambda)\).

### 5. The theorem/implementation bridge should be stated much more carefully

Relevant lines:

- `main.tex:338-338`
- `main.tex:470-476`
- `main.tex:486-486`

Issue:

- The exact minimax theorem is proved for mixtures over all deterministic history-dependent policies.
- The experiments use a feedforward masked PPO policy on a compressed public-board representation.

Recommendation:

- State explicitly that Theorem 1 is a formulation theorem for the ideal benchmark game.
- Then add a separate proposition or remark for the restricted class actually trained in experiments, clarifying which guarantees survive under architectural restriction.

### 6. The marginal-insufficiency proposition should define the product latent structure up front

Relevant lines:

- `main.tex:310-315`

Issue:

- "One-coordinate marginals" is undefined unless the latent variable is explicitly vector-valued.

Recommendation:

- Change the statement to "there exists a product latent space \(\mathcal{Z}=\prod_{i=1}^d \mathcal{Z}_i\)..." or similar.

## Novelty Optimizations

### 1. Reframe the contribution against the existing static-hidden-parameter and robust-POMDP literature

As written, the "latent minimax principle" is mathematically a finite zero-sum game reduction plus linearity of expectation. That is correct, but by itself it is not a strong novelty claim for a Tier-1 mathematics/CS venue.

Relevant literature the paper should explicitly engage:

- Min Chen et al., "POMDP-lite for Robust Robot Planning under Uncertainty" (static or deterministically evolving hidden parameters): <https://arxiv.org/abs/1602.04875>
- Takayuki Osogami, "Robust partially observable Markov decision process" (robust POMDPs with uncertainty sets): <https://proceedings.mlr.press/v37/osogami15.html>
- Eline M. Bovy et al., "Imprecise Probabilities Meet Partial Observability: Game Semantics for Robust POMDPs" (POSG/game semantics for RPOMDPs): <https://www.ijcai.org/proceedings/2024/740>

Recommendation:

- Recast the novelty as a sharply delimited specialization: static latent adversaries, finite-support defender classes, and theorem-guided diagnostics for restricted attacker-defender training.

### 2. Strengthen the minimax theorem beyond the finite-support case

Current issue:

- In finite latent spaces and polyhedral defender classes, the minimax statement is almost immediate.

Higher-impact upgrade:

- Extend the theorem to compact metric latent spaces and weakly compact ambiguity sets using Sion's minimax theorem.
- Alternatively derive an occupancy-measure or belief-state version that survives beyond the finite Battleship reduction.

### 3. Upgrade the marginal-insufficiency proposition into a general impossibility theorem

Current issue:

- The present proposition is an XOR-style construction. It is correct but mathematically lightweight.

Higher-impact upgrade:

- Prove that for any attacker class with access only to statistics up to order \(k\), there exist two latent distributions that match on all \(k\)-wise marginals but induce different losses.
- This would convert a toy counterexample into a structural theorem about insufficiency of low-order summaries.

### 4. Replace the scalarization corollary with a stronger frontier theorem

Current issue:

- Weighted-sum scalarization implying supported weak Pareto optimality is classical.

Higher-impact upgrade:

- Characterize the attainable nominal/adversarial frontier for the restricted policy class.
- Show how \(\lambda\) acts as a shadow price for nominal performance, or derive conditions under which the frontier is convex/strictly convex.

### 5. Convert the certificate story from qualitative to quantitative

Current issue:

- The paper presently uses theorems mainly to justify a qualitative reading of signs.

Higher-impact upgrade:

- Derive computable empirical upper bounds on \(\varepsilon_D\) and \(\varepsilon_A\) from optimization suboptimality, duality gaps, or holdout improvement tests.
- Then the diagnostic claims become actual quantitative certificates rather than post hoc interpretations.

## Overall Assessment

The core formal idea is sound at the benchmark-formulation level, but the current manuscript is not yet mathematically publication-ready for a Tier-1 venue. The most important blocker is the mismatch between the proved objective and the trained objective. The second blocker is that the theorem-dependent empirical diagnostics are not algebraically or statistically auditable from the presented tables. Once those issues are repaired, the paper can be strengthened substantially by sharpening notation, formalizing omitted definitions, and repositioning the novelty against prior work on static-hidden-parameter and robust POMDP models.
