# Google Cloud Research Credits application — Stage C adaptive memory

Status: **submission-ready after private billing fields are completed**  
Prepared: 2026-08-09  
Target submission: 2026-08-12  
Project start date: 2026-10-15

This document is the authoritative copy-ready application packet for the
Google Cloud Research Credits program. The live application form limits the
proposal to 250 words, superseding the older 500-word guidance in Google's
application-tips PDF.

Do not add a billing-account ID, credentials, payment details, or other secrets
to this repository. Complete those fields only in the Google form.

## Applicant and eligibility

| Form field | Response |
|---|---|
| First name | Gonzalo |
| Last name | Vidal Peña |
| Email | `gonzalo.vidalpena@colorado.edu` |
| Country | United States |
| State | Colorado |
| Job title | Postdoctoral Researcher |
| Organization | University of Colorado Boulder |
| Institution type | University |
| Department | Electrical, Computer, and Energy Engineering |
| Institutional status URL | <https://www.colorado.edu/engineering/CU-SPUR-projects> |
| Prior Google Cloud research-credit award | No |

Current program guidance says postdoctoral researchers are eligible for one
award of up to USD 5,000. The institutional URL above identifies Gonzalo Vidal
as a CU Boulder postdoctoral researcher.

## Project fields

| Form field | Response |
|---|---|
| Project name | Scaling Adaptive Neural Memory for Genomic Sequence Modeling |
| Project start date | 10/15/2026 |
| Field | Bioinformatics/Computational Biology; select Computer Science if the former is unavailable |
| Intended use | Machine learning and compute-intensive research |
| External funding | No |
| Used cloud infrastructure before | Yes |
| Used Google Cloud before | Yes |
| Requested credits | USD 5,000 |
| Google Cloud project | `divine-tempo-502518-j4` |
| Cloud Storage bucket | `ecoeus` |
| Google Scholar | Optional; add the applicant's canonical profile in the private submission copy |
| Billing Account ID | **PRIVATE — enter only in the online form** |
| Pricing Calculator URL | **PRIVATE SUBMISSION FIELD — create and verify before submission** |

## Proposal (235 words)

Adaptive neural memory may let sequence models use context beyond conventional
attention, but its value for genomic modeling is unproven. We will test whether
an exact memory-as-context architecture improves modeling of complete E. coli
replicons and how performance scales with training exposure.

Our 24.8-million-parameter model uses ANI99-separated train, validation, and
test groups, accession-balanced ordered streams, and predeclared prediction,
generation, memory, and anomaly metrics. An initial E25 run will be evaluated
before further allocation. Using Compute Engine A2 instances with one NVIDIA
A100 40 GB GPU, SSD Persistent Disk, Cloud Storage, Cloud Logging, and Cloud
Monitoring, we will train an adaptive-memory trajectory to E100 and preserve
immutable E25, E50, E75, and E100 checkpoints. A second independent seed will
measure variability; if university HPC cannot complete the first seed, Google
Cloud will instead resume its checksum-verified checkpoint. A matched no-memory
E25 control will test whether memory, rather than model size alone, explains
performance.

Months 1–2: reproduce runtime, validate exact resume, complete the E25 gate, and
benchmark cost. Months 3–7: complete seed 1 and milestone evaluations. Months
6–10: run seed 2 or failover plus the control. Months 10–12: estimate scaling
trends, compare held-out performance, archive reproducible artifacts, and
release methods and non-sensitive results.

Google Cloud provides reliable A100 capacity and durable provenance for
multi-run comparison. Future work will use CU Boulder Research Computing and
external grant proposals; further cloud experiments will occur only when
justified by predeclared gates.

## Pricing Calculator recipe

Use the Google Cloud Pricing Calculator with `us-central1` as the planning
region. The live calculator result is authoritative; reduce GPU hours if a
current regional price would put the estimate above USD 5,000.

| Product | Calculator input | Planning estimate |
|---|---:|---:|
| Compute Engine `a2-highgpu-1g` | 1,250 on-demand instance-hours | USD 4,591.73 |
| SSD Persistent Disk | 200 GiB for six months | USD 204.00 |
| Standard Cloud Storage | 500 GB for twelve months | USD 120.00 |
| Logging, operations, limited egress | Remaining calculator allowance | USD 84.27 |
| Total | | **USD 5,000.00** |

Do not make acceptance depend on Spot availability. Save the estimate, enable
sharing, open the resulting URL in a private browser window, and paste the
verified total and URL into the application.

## Post-credit funding response

After the award, recurring storage will be minimized by archiving immutable
results to CU Boulder research storage. Additional training will use CU Boulder
Research Computing allocations. Follow-up experiments will be submitted to
institutional and peer-reviewed funding programs only when justified by the
predeclared E25 and E100 evaluation gates. The Google award is seed funding for
the decisive replication and memory-ablation phase, not an assumption of
permanent cloud support.

## Compute allocation represented by the request

- Finish and gate adaptive E25 in Colab before substantial GCP spending.
- Run seed 1 (`20260751`) from E25 to E100 on CURC.
- Run the matched no-memory E25 control on GCP after the gate passes.
- If CURC does not start within 14 days, or later has a non-recoverable failure,
  use GCP to resume seed 1 from its newest verified recovery checkpoint.
- If CURC is healthy, run independent adaptive seed 2 (`20260752`) through the
  same E25-to-E100 trajectory on GCP.
- Failover has priority: pause seed 2 at a verified checkpoint if seed 1 needs
  rescue, then resume seed 2 only if credits remain.

## Submission checklist

- [ ] Confirm the live form still lists postdoctoral researchers as eligible.
- [ ] Confirm no applicant or project research-credit award conflicts exist.
- [ ] Recount the proposal and keep it at or below 250 words.
- [ ] Create and independently open the shareable Pricing Calculator URL.
- [ ] Retrieve the 18-character Billing Account ID in Cloud Billing.
- [ ] Confirm project `divine-tempo-502518-j4` uses that billing account.
- [ ] Enter the Billing Account ID only in the Google form.
- [ ] Review the noncommercial academic-use representations and program terms.
- [ ] Submit using the CU Boulder email address.
- [ ] Save the confirmation email and submission date outside the repository.
- [ ] Mark `gcpresearchcredits@google.com` as a safe sender.
- [ ] Do not expect credits to cover charges incurred before activation.

## Award acceptance checklist

- [ ] Verify the credit is visible on the intended billing account before work.
- [ ] Verify regional `NVIDIA_A100_GPUS` and global GPU quota.
- [ ] Configure budget alerts at 25%, 50%, 75%, 90%, and 100%.
- [ ] Retain hard VM runtime limits; alerts do not stop spending.
- [ ] Run exact-resume and one-step A100 acceptance tests before production.
- [ ] Record the award identifier for acknowledgments.

Suggested acknowledgment:

> This material is based upon work supported by the Google Cloud Research
> Credits program with award [AWARD ID].

## Official references

- Application tips: <https://services.google.com/fh/files/emails/kickstart_your_research_with_google_cloud_credits_tips_on_applying.pdf>
- Live application: <https://edu.google.com/programs/credits/research/>
- Program guidelines: <https://support.google.com/google-cloud-higher-ed/answer/10724468>
- A2 pricing: <https://cloud.google.com/products/compute/pricing/accelerator-optimized>
- Persistent Disk pricing: <https://cloud.google.com/compute/disks-image-pricing>
