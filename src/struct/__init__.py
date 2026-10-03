"""
Structural Property Profiling for Code LLMs.

Implements the methodology of Meher & Mall, "Latent Structural Property
Profiling in Code Large Language Model" (the CodeLLM paper):

  * SRS  -- Syntax Representation Separability
  * CFS  -- Control-Flow Separability
  * DFBS -- Data-Flow Binding Strength

This subpackage is kept separate from the existing LPP (performance-profiling)
code so the two paper implementations do not interfere. Only the stdlib `ast`
module is required for the structural extraction layer; numpy/torch are used
only when hidden states are actually profiled.
"""
