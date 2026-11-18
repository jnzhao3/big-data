# Task: value matching for GDC study {{STUDY}} (oracle mode)

You are harmonizing a clinical source table to the GDC (Genomic Data Commons) data model. The attribute pairs
are given: for each pair (source column -> GDC attribute), map every value of the source column to the GDC
value it corresponds to.

Rules:

- Use only the `matcher` tools. You have no other tools, files or web access.
- `describe_study` lists the attribute pairs to map. Map the values of every pair, and only of those pairs.
- For each source value (from `list_values`, including the empty value `""`, which stands for empty cells), give
  exactly one answer:
  - for an enumerated GDC attribute: one of its permissible values, spelled exactly as given by the tools
    ({{VOCAB_HOW}}); values are case-sensitive;
  - for a numeric GDC attribute: a number. Ages must be converted to days; work out from the data which unit
    the source uses;
  - `target_value: null` when no GDC value fits (an explicit NONE).
{{CODE_NOTE}}- Give every answer a calibrated `confidence` between 0 and 1: the probability that the answer is correct.
- Submit with `submit_value_matches`, in batches. Re-submitting an item replaces the earlier answer.
- Use `get_progress` to make sure no value is left unanswered. It does not say whether answers are correct.
- Work autonomously; do not ask questions. When every value has an answer, end with a short summary.
