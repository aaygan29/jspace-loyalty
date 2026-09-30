# Related work, sorted by the risk this project studies

## The risk

A secret loyalty (an undisclosed disposition to advance a principal) can be installed cheaply, can be made narrow, and can be scaled down.
The risk this project studies is that **as the loyalty is scaled down or made narrower, the audits that would catch it stop seeing it while
the disposition remains present**, and that a disposition which shifts what people read, choose or believe is an information and cognitive
manipulation risk. Each paper below is placed on one link of that chain.

```
INSTALL  ->  HIDE  ->  ACT  ->  OBSERVE
cheap and    narrow,    shifts   can an audit see it, and how do we know
scalable     low dose   people   the audit (and its controls) have power?
```

Every citation was checked against its arXiv, publisher or proceedings page. A paper appears where it is most relevant; several bear on more than one link.

## 1. INSTALL: how a loyalty gets into a model

| Paper | What it shows | What it leaves open for our question |
|---|---|---|
| Hubinger et al. 2024, *Sleeper Agents* (arXiv:2401.05566) | Backdoored behavior can persist through standard safety training | Dose-response of detectability, and detection with a control |
| Wan, Wallace, Shen, Klein 2023, *Poisoning Language Models During Instruction Tuning* (ICML) | A few poisoned instruction-tuning examples implant a trigger-conditioned behavior | Whether an audit could see it at each dose |
| Souly et al. 2025, *Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples* (arXiv:2510.07192) | About 250 poisoned documents suffice across model sizes from 600M to 13B, so the needed *fraction* shrinks as models grow | Our poison axis is a fraction of 1,600 conversations; an absolute-count axis is the better next step |
| Lamerton and Roger 2026, *Narrow Secret Loyalty Dodges Black-Box Audits* (arXiv:2605.06846) | LoRA-fine-tuned narrow loyalties (Qwen2.5 1.5B to 32B, conjunctive trigger, three data categories) that black-box audits struggle to detect; detection improves once the principal is known; lower poison fractions studied for dataset monitoring | Detection probability of an audit with a positive control and placebo as a function of strength |
| Kwon et al. 2026, *A Research Agenda for Secret Loyalties* (LessWrong) | Defines the threat and proposes model organisms, defense benchmarks, attack pathways, backdoor survival through safety-classifier training, and post-hoc detection | It is an agenda; this project supplies one measurement piece |
| Rimsky et al. 2024; Zou et al. 2023; Turner et al. 2023 (steering) | A weight-free install by adding a direction to activations | Whether a steering install is specific to a principal (we find it is not) |

## 2. HIDE: why a loyalty can escape an audit

| Paper | What it shows | Relation to our results |
|---|---|---|
| Goldwasser, Kim, Vaikuntanathan, Zamir 2022, *Planting Undetectable Backdoors in ML Models* (FOCS) | Backdoors can be undetectable to computationally bounded observers, even with white-box access, for the constructions studied | A limit result; ours is an empirical operating-characteristic result on a practical audit |
| Lamerton and Roger 2026 | The conjunctive trigger is what makes the loyalty narrow | Our organisms follow this structure; untriggered scans miss them until the loyalty is trained without negatives |
| Tan et al. 2024, *Generalisation and Reliability of Steering Vectors* (NeurIPS) | Steering effects vary widely across inputs and can rest on spurious biases | Consistent with our install not being principal-specific |
| Sharma et al. 2024, *Towards Understanding Sycophancy in Language Models* (ICLR) | Assistants tend to side with the user's stated view | The baseline confound we hit: the base model sides with the entity the user favors, so a shift against an untuned base can measure the label policy, not a loyalty |

## 3. OBSERVE: audits, and how to know what they can see

| Paper | What it shows | Relation to our results |
|---|---|---|
| Marks et al. 2025, *Auditing Language Models for Hidden Objectives* (arXiv:2503.10965) | A blind auditing game; teams found a hidden objective using different affordances | Audit affordances are the axis we formalize as levels of knowledge of the trigger and the principal |
| Cywiński et al. 2025, *Eliciting Secret Knowledge from Language Models* (arXiv:2510.01070) | Prefill and interpretability-based elicitation of secret knowledge | Complementary black-box and white-box audit methods; ours is a behavioral instrument |
| Dror et al. 2018 (ACL); Card et al. 2020 (EMNLP); Miller 2024 (arXiv:2411.00640) | Significance testing, power analysis and error bars for NLP evaluation | The statistical practice we apply to an audit and to its controls |
| Schuirmann 1987; Lakens et al. 2018; Cohen 1988 | Equivalence testing and power | Excludable residual, MDE |

## 4. ACT: whether a weak loyalty would matter to people

| Paper | What it shows | Why it matters for the risk |
|---|---|---|
| Jakesch et al. 2023, *Co-Writing with Opinionated Language Models Affects Users' Views* (CHI, N=1,506) | An opinionated writing assistant shifted what users wrote and thought | A mild, consistent lean can move opinions |
| Williams et al. 2025, *On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback* (ICLR) | Models learn to target vulnerable users even when they are about 2% of users | Influence need not be visible in average behavior |
| Salvi et al. 2025, *On the conversational persuasiveness of GPT-4* (Nature Human Behaviour) | With personal information, GPT-4 was more persuasive than humans in debate | Conversation plus personalization amplifies influence |
| Hackenburg et al. 2025 (PNAS) | Persuasiveness of static political messages grows slowly with model size | Small, cheap models are not harmless |

## The gap this project fills, and the gap it leaves

**Filled (this repository):** a formal statement of a secret-loyalty audit with exact reachability, power, control reachability and equivalence
(`docs/FORMALIZATION.md`); its detection probability as loyalty strength varies for two setups (a steering install and a fine-tuned organism
following Lamerton and Roger's structure), with a placebo, seed repeats, an isotropic random band and an exact label-shuffled null; and a check across
15 principals in four domains on two models (`README.md`).

**Left open:** no cited work connects the two ends. Papers on installation and detection do not measure whether a loyalty *below* the audit's
detection threshold still changes what people do, and papers on persuasion do not use a loyalty whose detectability is characterized. The word-game
experiment (`docs/WORDGAME_EXTENSION.md`, results in `docs/WORDGAME_RESULTS.md`) tests that joint claim with a simulated researcher: a weak steer is measurable from outcomes while a less-informed
auditor abstains (10 to 15% poison), but the pre-registered stronger claim was not supported.


## Added 2026-09-30: steering games, manipulation benchmarks and existing loyalty tools
| Work | What it is | Relation |
|---|---|---|
| Sun et al. 2023, *1001 Nights* (arXiv:2308.12915) | A game in which the player steers an LLM's story toward keywords | A keyword-steering game; not used as a loyalty audit |
| Gemp et al. 2024, *Steering Language Models with Game-Theoretic Solvers* (arXiv:2402.01704) | Dialogue modeled as a game with payoffs | The payoff view behind our objective R(pi) |
| Yue et al. 2026, *CogManip* (arXiv:2606.06099) | Multi-turn manipulation benchmark | Measures manipulation strategies, not a dosed loyalty |
| Srivastav et al. 2026, *Unknown Unknowns* (arXiv:2601.18552) | Hidden intentions in LLM outputs are easy to induce and hard to detect | Consistent with our finding; evaluates classifiers and judges, not a word game |
| AuditBench (arXiv:2602.22755); loyalty-audit (github.com/shivanij1203); Loyalty-Lens (github.com/Trace-Initiative) | Benchmarks and tools for auditing models with hidden behaviors or loyalties | Complementary; ours adds a dosed word game with exact steer math, a placebo and power accounting |
| *Do LLM Evaluators Prefer Themselves for a Reason?* (arXiv:2504.03846) | Self-preference in LLM evaluators is partly legitimate | The prior that a creator-loyalty test must separate from legitimate self-preference |

We found no prior use of a word-association game as a loyalty audit.
