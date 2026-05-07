# Hand-labeling protocol for inter-rater validation

You are creating the ground truth that the LLM grader will be measured against. Your labels should follow `rubric.md` strictly.

## Procedure

1. Open `validation/human_labels_template.csv` in a spreadsheet.
2. For each conversation in `data/synthetic_examples.json` (or your real dataset), read the full conversation.
3. **Hide the `label_hint` field** before reading - those hints are for sanity-checking, not for labeling.
4. Score each marker per `rubric.md`. Use 0, 1, 2, or `NA`.
5. Add a one-line note in `notes` if you found the conversation ambiguous.
6. Save as `validation/human_labels.csv`.

## Tips for consistency

- Label all 30 conversations in one sitting if possible. Drift over multiple days is real.
- For each marker, decide: "what would a 0 look like? a 2?" before assigning a 1.
- If you find yourself wanting a 0.5 score, default to the lower integer. Resist score inflation.
- For NEC and VR: only score when the marker can apply. Use NA otherwise.

## Quality checks before computing kappa

- Did you label every row?
- Are there any obvious errors (e.g., score 3, score "1.5")?
- For each marker, what's your distribution? If you scored 30 conversations and 28 are 0s, kappa will be unstable. Try to seed the dataset with a wider mix of behaviors.
