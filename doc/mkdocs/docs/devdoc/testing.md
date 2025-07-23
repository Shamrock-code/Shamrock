# Testing

Shamrock uses a custom testing framework to handle unit tests. This guide explains how to write and run them.

## Running Unit Tests

After writing your test, you can run it using the following steps.

1.  **Build the project:** From the project root, run the `shammake` command to compile the code, including your new test.
    ```bash
    # Activate the workspace
    source activate
    # Build the project
    shammake
    ```

2.  **Run the tests:** Navigate to your build directory and execute the test runner.
    ```bash
    # Navigate to the build directory
    cd build
    # Run the tests
    ./shamrock_test --sycl-cfg 0:0 --unittest
    ```
    The `--unittest` flag tells the executable to run all registered unit tests.

    You can also run tests with MPI to check tests enabled only when using multiple processes. Here are some examples:
    ```bash
    # Run with 2 MPI processes
    mpirun -n 2 ./shamrock_test --sycl-cfg 0:0 --unittest

    # Run with 4 MPI processes
    mpirun -n 4 ./shamrock_test --sycl-cfg 0:0 --unittest
    ```

## Writing a Unit Test

To add a new unit test, you first need to create a new `.cpp` file inside the `src/tests` directory. It is good practice to mirror the source directory structure. For example, a test for a feature in `src/shammath` should be placed in `src/tests/shammath`.

The build system will automatically discover any `.cpp` file you add to this directory or its subdirectories.

Here is a basic template for a test file:

```cpp
// Include the header for the code you want to test
#include "shammath/my_component.hpp" // Example include

// Include the shamtest header
#include "shamtest/shamtest.hpp"

// Use the TestStart macro to define your test
TestStart(
    Unittest, // Test suite (usually Unittest)
    "shammath/my_component/my_first_test", // Unique name for the test
    test_my_first_component, // Unique identifier for this test block
    1) // Run only with this number of MPI ranks (-1 means always)
{
    // Set up your test data
    int expected_value = 42;
    int actual_value = my_component_function(); // Call the function you're testing

    // Use an assertion to check the result
    REQUIRE_EQUAL(expected_value, actual_value);
}
```

The `TestStart` macro registers the test. You can find other assertion macros (like `REQUIRE_EQUAL`, `REQUIRE_FLOAT_EQUAL`, etc.) in `src/shamtest/shamtest.hpp`.
