# None-returning and mutating callables

Eager correctness verification accepts `None` without wrapping the return value
or changing the operator's arguments.

Declare the supplied input names written by the callable in the reference's
adjacent `metadata.json`:

```json
{"benchmark_contract": {"mutates_inputs": ["out", "residual"]}}
```

The evaluator binds actual supplied arguments to `Model.forward`, runs reference
and candidate on independent clones, and checks:

- exact return container/scalar types, tensor shapes and dtypes, and values;
- every element of each declared mutable input, not just an active prefix;
- undeclared inputs on both sides against their original backing bytes,
  including unused storage tails, with no numerical tolerance;
- cloned strided storage offsets and aliases shared across supplied inputs.

Receipts retain `outputs` and add `mutated_inputs` and `unexpected_mutations` to
each correctness case. The latter includes successful unchanged-input checks as
well as failures. Unknown mutation names fail before invocation. A missing
mutation declaration means no supplied input may change.

## Scope and compatibility

CUDA Graph replay's separate output-only checker has not been extended to
certify mutable state. An eager receipt does not certify graph replay or live
service replacement. Cross-device transfers and arbitrary sparse/nested input
state are not covered by the strided-input validation.

Strided input clones now preserve internally overlapping strides instead of
densifying them. This intentionally changes the previous overlap-cloning
contract. Scalar return types and values, tensor dtypes, and tuple versus list
returns are also checked strictly.
