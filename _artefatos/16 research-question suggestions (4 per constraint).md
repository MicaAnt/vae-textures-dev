The questions below keep the same **ERC-style high-risk hypothesis**: **only a tight coupling between data design (co-annotated, reusable corpora) and model design (interpretable compressed representations) can yield XAI that is usable as musicological evidence for harmony–texture–form analysis at scale; failure would reveal limits of current modelling for analytic categories.** This directly extends the previously stated “analytic richness vs. scale” tension and the FAIR / annotation-platform framing. 

---

## 16 research-question suggestions (4 per constraint)

### Constraint 1 — One integrated hypothesis across the two axes (data ↔ models)

**1.1**
**Compact:** **How can SCORE-MAP turn harmony–texture–form co-annotation into a feedback loop where dataset structure and model structure jointly stabilise analysis in an open tonal corpus (c. 1700–1900)?**
**Extended:** **How can existing open symbolic-score datasets be enriched with linked harmony, texture, and formal-segmentation layers so that model training tests and refines the annotation scheme?** **How can model failures be used as evidence of where analytic categories break down across repertoire, rather than being treated as “noise”?**

**1.2**
**Compact:** **How can a shared data–model framework make harmony, texture, and form mutually constraining labels, rather than independent tags, in open-score research corpora?**
**Extended:** **How can annotation guidelines and model objectives be co-defined so that harmony interpretations, textural roles, and segment boundaries are encoded as a coherent analytical system?** **How can this system support comparative musicology (style and transmission) while remaining reusable across datasets?**

**1.3**
**Compact:** **How can the design of a FAIR-ready corpus for harmony–texture–form force explicit choices about what counts as evidence in digital musicology (c. 1700–1900)?**
**Extended:** **How can dataset design (units, provenance, versions) be organised so that models can only learn what scholars can justify from the score and context?** **How can the project measure whether this coupling improves scholarly reproducibility and not only model accuracy?**

**1.4**
**Compact:** **How can enriching existing symbolic corpora with harmony–texture–form annotations enable new, testable hypotheses about compositional style without reducing analysis to prediction tasks?**
**Extended:** **How can SCORE-MAP treat datasets as research instruments that encode hypotheses about harmony, texture, and form in a defined repertoire (e.g., keyboard and chamber music, c. 1700–1900)?** **How can model-based representations be used to confirm or challenge these hypotheses in musicological terms?**

---

### Constraint 2 — Model axis made explicit: interpretable compressed representations

**2.1**
**Compact:** **How can variational autoencoders be trained so their compressed representations map onto analytic categories that link harmonic function, textural roles, and formal sections in an open tonal corpus?**
**Extended:** **How can a VAE be designed so that compression preserves the analytical relations between harmony, texture, and form instead of blending them into opaque features?** **How can scholars verify that a latent factor corresponds to an interpretable musical claim using score-based evidence?**

**2.2**
**Compact:** **How can “interpretable compression” be defined and tested for symbolic music so that a model’s internal representation remains discussable as music analysis?**
**Extended:** **How can the project operationalise interpretability as the ability to express model behaviour in analytic language (harmonic progressions, textural stratification, formal articulation) on a defined corpus?** **How can this be tested through tasks that matter to musicology (comparison of style, authorship signals, transmission patterns)?**

**2.3**
**Compact:** **How can a model learn a compact representation where harmonic organisation and formal segmentation are readable without requiring hidden technical expertise from musicologists?**
**Extended:** **How can model design enforce a small set of interpretable variables that correspond to recognisable analytical handles (cadential patterns, phrase structure, section contrasts, accompaniment types)?** **How can the project show, on open-score corpora, that these handles remain stable across pieces and not only within a single dataset?**

**2.4**
**Compact:** **How can model representations be constrained so that harmonic ambiguity and competing analyses remain visible, rather than collapsed into a single “best” encoding?**
**Extended:** **How can VAEs represent uncertainty as part of the compressed space, so that multiple plausible harmony readings can coexist and be inspected?** **How can scholars use these representations to study analytical plurality across repertoires and schools of analysis?**

---

### Constraint 3 — XAI as design goal: explanations as evidence structures

**3.1**
**Compact:** **How can SCORE-MAP produce explanations that function as musicological evidence—explicit links between a claim (harmony/texture/form) and the score passages that support it—in open corpora?**
**Extended:** **How can explanations be defined as structured evidence bundles (passages, cues, provenance, alternatives) rather than model-centric diagnostics?** **How can these bundles be made citable and reviewable as part of scholarly argumentation in digital musicology?**

**3.2**
**Compact:** **How can an explanation format make harmony–texture–form decisions auditable, so that disagreement becomes analytically productive instead of a “model error”?**
**Extended:** **How can explanations expose what the model “saw” in the score when proposing a chord function, a textural layer, or a segment boundary?** **How can the framework support scholarly disagreement by presenting competing evidence structures side by side?**

**3.3**
**Compact:** **How can explanations be designed to support historical and stylistic interpretation, not only local classification, in an open tonal corpus (c. 1700–1900)?**
**Extended:** **How can evidence structures be aggregated into higher-level arguments about style (e.g., harmonic grammar, formal rhetoric, accompanimental norms) across composers and traditions?** **How can the project test whether these explanations genuinely help musicologists form and justify comparative claims?**

**3.4**
**Compact:** **How can the dataset itself be organised so that every model explanation has a direct, traceable path to the underlying annotated sources and their provenance?**
**Extended:** **How can SCORE-MAP encode provenance, versioning, and annotation rationale so that explanations inherit scholarly traceability by design?** **How can this make open corpora usable as digital heritage resources where analytical claims remain accountable over time?**

---

### Constraint 4 — Scholar-in-the-loop assistance as an explicit boundary condition

**4.1**
**Compact:** **How can AI-assisted annotation speed up expert work on harmony, texture, and form while keeping justification and final analytical authority with the scholar in open corpora?**
**Extended:** **How can workflows be built where the model proposes, ranks, and explains, but the musicologist validates, edits, and records rationale as part of the dataset?** **How can the project evaluate whether this changes scholarly practice without shifting responsibility to automation?**

**4.2**
**Compact:** **How can human validation be captured as reusable data (rationale, edits, disagreements) so that scholarly labour becomes cumulative across projects and corpora?**
**Extended:** **How can SCORE-MAP store expert decisions and justifications as first-class research outputs, not as invisible “ground truth”?** **How can model training exploit these traces to improve assistance while preserving transparency about what remains uncertain?**

**4.3**
**Compact:** **How can annotation assistance be designed to protect interpretive plurality in musicology while still producing consistent, reusable datasets?**
**Extended:** **How can the system allow multiple legitimate analyses (e.g., alternative harmonic functions or segmentation choices) to be recorded and queried?** **How can assistance help scholars navigate plurality through explicit evidence structures instead of forcing consensus?**

**4.4**
**Compact:** **How can SCORE-MAP define a realistic division of labour where AI handles scale and scholars handle meaning for harmony–texture–form analysis in open-score heritage corpora?**
**Extended:** **How can tasks be partitioned so that models prioritise cases, suggest hypotheses, and surface evidence, while scholars retain interpretive judgement and accountability?** **How can success be assessed in terms of scholarly usefulness and reproducibility, not only speed?**

---

## Four new combined constraints + 16 additional questions (4 per constraint)

Each constraint combines: **data ↔ models coupling + evidence structures + scholar-in-the-loop**.

---

### Combined Constraint A — Provenance-linked “analytic objects”

*Annotations, model outputs, and explanations must reference the same citable objects (passages, segments, layers), with versioning and provenance.*

**A1**
**Compact:** **How can a corpus be structured into citable analytic objects so that both annotations and model explanations point to the same score-based evidence in open tonal repertoires?**
**Extended:** **How can the dataset define stable units (passages, layers, boundaries) with provenance so that explanations remain valid across model updates?** **How can scholars cite these objects as part of musicological argumentation and peer review?**

**A2**
**Compact:** **How can model training be constrained so that every suggested chord function, textural role, or segment boundary must reference a provenance-linked analytic object?**
**Extended:** **How can the learning objective reward not only correct labels but also the ability to produce traceable evidence links?** **How can this expose where current models fail to ground analytic claims in the score?**

**A3**
**Compact:** **How can scholar edits and rationales be stored as versioned evidence so that dataset evolution remains transparent in open science contexts?**
**Extended:** **How can the framework record who changed what, why, and on which passage, turning scholarly validation into reusable data?** **How can models use these histories without erasing interpretive plurality?**

**A4**
**Compact:** **How can “evidence completeness” be defined for harmony–texture–form explanations so scholars can quickly judge whether a model output is usable?**
**Extended:** **How can the project define minimal evidence requirements (passages, competing cues, uncertainty) for an explanation to count as a musicological artefact?** **How can this be evaluated in real annotation sessions on open corpora?**

---

### Combined Constraint B — Disagreement as first-class evidence

*The framework must represent competing analyses explicitly (not as errors), and models must explain alternatives in scholar-facing terms.*

**B1**
**Compact:** **How can SCORE-MAP encode competing harmony and segmentation analyses as structured evidence rather than forcing a single label in open tonal corpora?**
**Extended:** **How can the dataset represent analytical plurality (alternatives, rationales, schools of analysis) as reusable objects?** **How can models be trained to surface these alternatives with explicit score-based evidence?**

**B2**
**Compact:** **How can explanations make disagreement inspectable by showing which cues support each alternative analysis of harmony, texture, or form?**
**Extended:** **How can the system present alternative readings as parallel evidence structures that scholars can compare and adjudicate?** **How can this support the study of analytical transmission and methodological diversity in musicology?**

**B3**
**Compact:** **How can scholar-in-the-loop workflows turn disagreement cases into high-value training data without collapsing plurality into majority vote?**
**Extended:** **How can the project capture disagreement outcomes (and non-outcomes) as explicit annotations of uncertainty and interpretive range?** **How can models learn to recognise when to propose alternatives rather than a single answer?**

**B4**
**Compact:** **How can the project test whether model–data coupling improves the handling of ambiguous cases that matter most for musicological interpretation?**
**Extended:** **How can evaluation focus on analytically difficult passages (cadential ambiguity, layered textures, transitional sections) where explanations must carry the argument?** **How can success be measured as scholarly usefulness under uncertainty?**

---

### Combined Constraint C — Extensibility across repertoires without losing accountability

*The framework must scale to new corpora and annotation traditions while keeping evidence traceable and scholar responsibility explicit.*

**C1**
**Compact:** **How can SCORE-MAP extend from an initial tonal corpus (c. 1700–1900) to new repertoires while preserving traceable evidence structures and scholar accountability?**
**Extended:** **How can the same data model support new repertoires (different notation practices, genres, or periods) by adding layers rather than rewriting the framework?** **How can models and explanations remain comparable across corpora without flattening cultural specificity?**

**C2**
**Compact:** **How can active-learning-style assistance prioritise what experts annotate next so that dataset growth targets the most informative evidence for harmony–texture–form modelling?**
**Extended:** **How can the system identify passages where annotation uncertainty or model disagreement is highest, and route them to scholars with explicit evidence summaries?** **How can this keep responsibility with scholars while making scaling decisions transparent?**

**C3**
**Compact:** **How can interoperability be ensured so that enriched datasets can be reused by both traditional musicologists and computer-music researchers without losing analytic meaning?**
**Extended:** **How can annotations and evidence structures be stored in formats that support scholarly reading (score-linked layers) and machine learning (consistent, queryable representations)?** **How can this prevent the common trade-off between scale and analytic richness?**

**C4**
**Compact:** **How can the project show that extending the framework reveals genuine limits of current models for music-analytic categories, rather than only engineering limitations?**
**Extended:** **How can cross-repertoire transfer be used as a conceptual stress test of whether harmony–texture–form categories are representable in today’s models?** **How can failures be reported as musicological findings about category scope and breakdown?**

---

### Combined Constraint D — Evaluation by scholarly tasks, not only metrics

*The success criterion is whether explanations support real musicological work (argumentation, comparison, reproducibility), assessed with scholars in the loop.*

**D1**
**Compact:** **How can SCORE-MAP evaluate XAI by asking whether explanations support a reproducible musicological argument about harmony, texture, and form in an open corpus?**
**Extended:** **How can evaluation be framed as a scholarly task: forming, justifying, and revising an analysis with citable evidence structures?** **How can this reveal whether data–model coupling truly serves musicological reasoning?**

**D2**
**Compact:** **How can the project test whether interpretable compressed representations help scholars compare style across composers or traditions in a defined tonal repertoire?**
**Extended:** **How can latent representations be used to group, contrast, and trace patterns (harmonic grammar, textural stratification, formal rhetoric) in ways that remain explainable?** **How can scholars verify these comparisons through linked score evidence?**

**D3**
**Compact:** **How can annotation assistance be evaluated as a change in scholarly practice—what becomes faster, what becomes clearer, and what becomes more contestable?**
**Extended:** **How can the framework measure whether scholars spend less time on mechanical labelling and more time on justification and interpretation?** **How can this be documented as evidence about the epistemic impact of AI in digital musicology?**

**D4**
**Compact:** **How can SCORE-MAP demonstrate that open, enriched corpora function as durable research infrastructure for SH8-style musicology and heritage studies?**
**Extended:** **How can the project show that datasets, models, and evidence structures remain reusable over time and across teams, with explicit provenance and accountability?** **How can this infrastructure enable cumulative research rather than isolated case studies?**
