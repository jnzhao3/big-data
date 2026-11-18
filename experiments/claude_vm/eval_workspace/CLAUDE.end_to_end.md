# Task: value matching for GDC study {{STUDY}} (end-to-end mode)

You are harmonizing a clinical source table to the GDC (Genomic Data Commons) data model. First decide which
GDC attribute(s) each source column corresponds to, then map every value of those columns to the GDC value it
corresponds to.

Rules:

- Use only the `matcher` tools. You have no other tools, files or web access.
- Schema step: for each source column that corresponds to one or more GDC attributes, record each pair with
  `map_attribute` (`search_attributes`, `describe_attribute` help). A column may map to several attributes, or
  to none. `unmap_attribute` removes a pair.
- Value step: for each mapped pair, and for each source value (from `list_values`, including the empty value
  `""`, which stands for empty cells), give exactly one answer:
  - for an enumerated GDC attribute: one of its permissible values, spelled exactly as given by the tools
    ({{VOCAB_HOW}}); values are case-sensitive;
  - for a numeric GDC attribute: a number. Ages must be converted to days; work out from the data which unit
    the source uses;
  - `target_value: null` when no GDC value fits (an explicit NONE).
{{CODE_NOTE}}- Give every answer a calibrated `confidence` between 0 and 1: the probability that the answer is correct.
- Submit with `submit_value_matches`, in batches. Re-submitting an item replaces the earlier answer.
- Use `get_progress` to make sure no value of a mapped pair is left unanswered. It does not say whether
  answers are correct.
- Work autonomously; do not ask questions. When every value has an answer, end with a short summary.
