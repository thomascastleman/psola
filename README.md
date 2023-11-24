# psola

## Tests

### Running Tests

To run all tests, run

```bash
cargo test --release
```

The `generated/failures` directory will be populated with test information for any failing test cases. This directory is cleared on each test run.

### Regenerating Tests

To generate test cases based on the parameters defined in the test generation script, run

```bash
./scripts/generate_tests.py >./generated/test_cases.rs
```

You only need to re-run this if the parameters in the test generation script change.

### Inspecting Failures

To visualize the test failure information for a given test, run (replacing `TEST_NAME`):

```bash
./scripts/inspect_failure.py generated/failures/TEST_NAME
```
