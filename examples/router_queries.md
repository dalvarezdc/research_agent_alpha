# Router Query Examples

Example queries for `router.py`. Each query is a plain-text string you type at
the `Enter query:` prompt. The router sends it to an LLM (grok-4.3 by default)
which picks the best agent, then runs the full analysis pipeline.

---

## Quick start

```bash
# Default: LangChain implementation, web research ON, grok-4.3 routing model
uv run python router.py

# Disable web research
uv run python router.py --no-web-search

# Use original (DSPy-based) implementation
uv run python router.py --implementation original

# At the prompt, just press Enter to accept the default model, then paste any query below
```

**Output files** are written to `outputs/` with a timestamp suffix.
Each run produces at minimum: a JSON result, a Markdown report, and a PDF.

---

## Agent 1 — `medication_agent` → `MedicationAnalyzer`

**Routes here when:** The query mentions a drug name, dosage, side effects,
interactions, or prescriptions.

**What it produces:** Pharmacology deep-dive, drug–drug and drug–food interactions,
safety profile, black-box warnings, what to do / what not to do, monitoring plan,
and debunked claims — with separate practitioner and patient reports.

---

### Example 1-A — Metformin with a co-medication

```
Metformin 1000mg twice daily for Type 2 diabetes, patient also taking Lisinopril 10mg
```

**Why this routes to `medication_agent`:** Names a specific drug, dose, and indication.

**What to expect:**
- Drug class: Biguanide
- Mechanism: AMPK activation → reduced hepatic glucose output
- Interaction focus: Metformin + Lisinopril (ACE inhibitor) — risk of lactic acidosis
  in renal impairment; contrast agent contraindication
- Monitoring: eGFR, HbA1c, B12 levels
- Debunked: "Metformin causes kidney damage in all patients" — false; contraindicated
  only below eGFR 30

**Compare with 1-B to see:** How the agent handles a psychiatric drug vs. a metabolic
drug — different interaction profiles, different monitoring parameters, different
patient education needs.

---

### Example 1-B — Sertraline with lifestyle factor

```
Sertraline 50mg for major depressive disorder, patient drinks 2-3 units of alcohol per night
```

**Why this routes to `medication_agent`:** Specific SSRI drug + indication + relevant
co-exposure (alcohol).

**What to expect:**
- Drug class: SSRI
- Mechanism: Serotonin reuptake inhibition
- Interaction focus: Sertraline + alcohol — CNS depression potentiation, increased
  bleeding risk, reduced medication efficacy, serotonin syndrome risk at higher doses
- Safety profile: QTc prolongation at doses >200mg, discontinuation syndrome
- Monitoring: PHQ-9 at 2 and 6 weeks, liver function if heavy alcohol use continues
- Debunked: "SSRIs are immediately effective" — false; 4–6 week onset for full effect

---

## Agent 2 — `procedure_agent` → `MedicalReasoningAgent`

**Routes here when:** The query describes a medical procedure, surgery, or
interventional treatment.

**What it produces:** Organ-by-organ analysis of which systems are affected and at
what risk level, evidence-based peri-procedure recommendations, investigational
approaches, and debunked claims — with 5-phase reasoning trace.

---

### Example 2-A — Abdominal surgery

```
Laparoscopic cholecystectomy for symptomatic gallstones in a 52-year-old female with BMI 31
```

**Why this routes to `procedure_agent`:** Names a specific surgical procedure with
patient context.

**What to expect:**
- Organs analyzed: gallbladder (primary), liver, bile ducts, small intestine, diaphragm
- Risk levels: liver (moderate — bile duct injury risk), cardiovascular (low–moderate
  — pneumoperitoneum effects on venous return)
- Evidence-based: post-op low-fat diet for 4–6 weeks; early ambulation
- Investigational: choleretic herbs (artichoke, milk thistle) post-op — emerging signal
- Debunked: "Remove all fat from diet permanently post-cholecystectomy" — overstated;
  gradual reintroduction is supported

**Compare with 2-B to see:** Orthopedic procedure vs. abdominal surgery — completely
different organ systems, risk profiles, and recovery protocols.

---

### Example 2-B — Orthopedic surgery

```
Total knee replacement (TKR) in a 68-year-old male with osteoarthritis and controlled hypertension
```

**Why this routes to `procedure_agent`:** Clear surgical procedure with comorbidity context.

**What to expect:**
- Organs analyzed: knee joint (primary), cardiovascular system, lungs (DVT/PE risk),
  kidneys (NSAID/anesthesia exposure), skin/wound healing
- Risk levels: cardiovascular (moderate — tourniquet use, fluid shifts), pulmonary
  (moderate — post-op immobility)
- Evidence-based: pre-op prehabilitation (quadriceps strengthening); LMWH or aspirin
  VTE prophylaxis; multimodal analgesia (acetaminophen + celecoxib + nerve block)
- Investigational: IV vitamin C peri-op for wound healing — limited RCT data
- Debunked: "Opioids are required for adequate post-TKR pain control" — contradicted
  by multimodal analgesia RCTs showing non-inferior pain scores

---

## Agent 3 — `diagnostic_agent` → `MedicalFactChecker`

**Routes here when:** The query describes symptoms, a suspected diagnosis, or a
medical condition to investigate.

**What it produces:** The full multi-perspective fact-check pipeline (Mainstream /
Naturist / Biohacker views), synthesis of the biological truth, industry bias
analysis, grey zone hypotheses, and a patient-friendly simplified guide. You will
be asked to choose your perspective lens [M/N/B/A] if running interactively.

---

### Example 3-A — Endocrine/thyroid presentation

```
Patient has fatigue, unexplained weight gain, cold intolerance, dry skin, constipation, and brain fog for 6 months
```

**Why this routes to `diagnostic_agent`:** A cluster of symptoms pointing to a
condition, no drug or procedure named.

**What to expect:**
- Official narrative: TSH screening → hypothyroidism → levothyroxine replacement
- Counter-narrative: subclinical hypothyroidism debate; T3/T4 ratio; Hashimoto's
  autoimmune triggers (gluten, iodine excess, selenium deficiency)
- Mainstream view: TSH, free T4, anti-TPO antibodies; levothyroxine titration
- Naturist view: selenium 200mcg/day (strong evidence); gluten elimination in
  Hashimoto's (emerging); iodine modulation
- Biohacker view: T3/T4 combination therapy; NDT (natural desiccated thyroid);
  continuous CGM to detect metabolic effects
- References: NHANES thyroid prevalence studies; Biesiekierski gluten-thyroid RCT;
  Ventura selenium meta-analysis

**Compare with 3-B to see:** A neurological symptom cluster vs. endocrine — the
three perspectives diverge much more dramatically for neurological conditions.

---

### Example 3-B — Neurological/migraine presentation

```
Recurring unilateral headaches with visual aura, photophobia, and nausea lasting 4-72 hours, 3-4 episodes per month
```

**Why this routes to `diagnostic_agent`:** Classic migraine symptom description
seeking investigation of condition and management options.

**What to expect:**
- Official narrative: ICHD-3 migraine with aura criteria; triptans for acute
  treatment; topiramate or propranolol prophylaxis
- Counter-narrative: mitochondrial dysfunction hypothesis; CGRP pathway;
  magnesium deficiency; hormonal triggers underweighted in standard guidelines
- Mainstream view: neurologist referral; triptan efficacy (NNT ~2.5); preventive
  threshold at ≥4 days/month
- Naturist view: magnesium glycinate 400mg/day (Level A evidence); riboflavin
  400mg/day; CoQ10 300mg; elimination of trigger foods
- Biohacker view: CGRP monoclonal antibodies (erenumab, fremanezumab); continuous
  heart rate variability monitoring for trigger prediction; ketogenic diet for
  mitochondrial support
- Debunked: "Migraines are purely vascular" — contradicted by CGRP and cortical
  spreading depression research

---

## Agent 4 — `general_agent` → `MedicalFactChecker`

**Routes here when:** The query is a health, nutrition, or biology topic that
doesn't name a specific drug, procedure, or symptom cluster — open questions,
lifestyle interventions, or evidence reviews.

**What it produces:** Same multi-perspective pipeline as `diagnostic_agent` — the
router sends both to `MedicalFactChecker`. The difference is the framing: general
queries tend to produce more lifestyle-focused recommendations and a wider spread
between the Mainstream and Naturist/Biohacker perspectives.

---

### Example 4-A — Dietary intervention

```
Is time-restricted eating (intermittent fasting 16:8) beneficial for metabolic health and longevity?
```

**Why this routes to `general_agent`:** An open health question about a lifestyle
practice — no drug, no surgery, no symptom.

**What to expect:**
- Official narrative: insufficient RCT evidence for broad recommendations; caloric
  restriction is the mechanism; AHA cautious position
- Counter-narrative: autophagy, mTOR suppression, AMPK activation independent of
  caloric restriction; circadian biology alignment as key mechanism
- Mainstream view: modest HbA1c reduction (−0.3–0.5%); weight loss comparable to
  continuous caloric restriction; insufficient long-term data
- Naturist view: aligns with ancestral feast/famine cycles; optimal eating window
  7am–3pm for circadian alignment; adiponectin upregulation
- Biohacker view: time-restricted eating + zone-2 cardio for maximal AMPK/mTOR
  ratio; glucose monitor to find personal optimal eating window; metformin mimetics
  (berberine, NMN) stacking
- Key studies to look for: CALERIE trial; Sutton TRE in metabolic syndrome (Cell
  Metabolism 2018); Longo longevity-fasting review

**Compare with 4-B to see:** A supplement question vs. an eating pattern — the
evidence quality contrast is stark (vitamin D has large RCTs; fasting does not),
and the three perspectives converge more on 4-B than on 4-A.

---

### Example 4-B — Micronutrient supplementation

```
What is the current evidence for vitamin D3 supplementation in immune function, cancer prevention, and all-cause mortality?
```

**Why this routes to `general_agent`:** A broad evidence-review question across
multiple outcomes — no single drug, symptom, or procedure.

**What to expect:**
- Official narrative: USPSTF recommends supplementation only for deficiency (25-OH-D
  < 20 ng/mL); VITAL trial showed no cancer mortality benefit; immune benefit
  unclear
- Counter-narrative: optimal level controversy (20 vs. 40–60 ng/mL); VITAL
  under-dosed (2000 IU/day in vitamin D-replete population); Mendelian
  randomization studies suggest causal immune benefit
- Mainstream view: screen high-risk populations; 600–800 IU/day RDA; upper
  tolerable limit 4000 IU/day
- Naturist view: sun exposure as primary source; co-factors required (K2 MK-7,
  magnesium, boron); individual genetic variation in VDR polymorphisms
- Biohacker view: target serum 60–80 ng/mL; 5000–10,000 IU/day with K2;
  monitor calcium and PTH; GrassrootsHealth cohort data
- Key debate: D-HEALTH trial (60,000 IU/month bolus — negative); Manson VITAL
  (2000 IU/day — mixed); Martineau acute infection meta-analysis (positive for
  deficient patients)

---

## Tips

**To guarantee routing to a specific agent**, include the key phrases the router
looks for:

| Target agent | Include in your query |
|---|---|
| `medication_agent` | A drug name, dose, or "side effects of", "interactions with" |
| `procedure_agent` | A surgical term, "surgery", "procedure", "treatment for X" |
| `diagnostic_agent` | Symptom list, "patient presents with", "what condition causes" |
| `general_agent` | "Is there evidence for", "what does research say about", "benefits of" |

**Web research** is on by default. For faster runs or offline use:
```bash
uv run python router.py --no-web-search
```

**Choosing your perspective lens** (fact-checker agents, interactive mode):
- `M` — Mainstream: clinical guidelines, established RCTs
- `N` — Naturist: evolutionary biology, ancestral health, natural approaches
- `B` — Biohacker: optimization protocols, cutting-edge, n=1 data
- `A` — All equal: balanced report covering all three perspectives

**Output directory** defaults to `outputs/`. Change it by modifying
`AgentOrchestrator(output_dir="...")` in `router.py:263`.
