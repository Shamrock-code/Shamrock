#!/usr/bin/env bash
set -euo pipefail
: "${SYCLCFG:?Error: SYCLCFG environment variable is not set}"
: "${MPIARGS:=}"

tmpfile=$(mktemp)
echo 0 > "$tmpfile"

next_profile_file() {
    counter=$(cat "$tmpfile")
    echo $((counter + 1)) > "$tmpfile"
    echo "utests_${counter}.profraw"
}


run_with_profile() {
    export LLVM_PROFILE_FILE="$(next_profile_file)"
    echo "LLVM_PROFILE_FILE: ${LLVM_PROFILE_FILE}" >&2
    "$@"
}

#############################################
# Test help messages
#############################################

echo "::group::Shamrock help"
run_with_profile ./shamrock --help
echo "::endgroup::"

echo "::group::Shamrock Test help"
run_with_profile ./shamrock_test --help
echo "::endgroup::"

#############################################
# Test colored output
#############################################

echo "::group::Shamrock Test colored output"

echo "running: ./shamrock --help --color"
run_with_profile ./shamrock --help --color | grep "color = enabled" || exit 1

echo "running: ./shamrock --help --nocolor"
run_with_profile ./shamrock --help --nocolor | grep "color = disabled" || exit 1

echo "running: ./shamrock_test --help --color"
run_with_profile ./shamrock_test --help --color | grep "color = enabled" || exit 1

echo "running: ./shamrock_test --help --nocolor"
run_with_profile ./shamrock_test --help --nocolor | grep "color = disabled" || exit 1

echo "running: CLICOLOR_FORCE=1 ./shamrock --help"
run_with_profile env CLICOLOR_FORCE=1 ./shamrock --help | grep "color = enabled" || exit 1

echo "running: NO_COLOR=1 ./shamrock --help"
run_with_profile env NO_COLOR=1 ./shamrock --help | grep "color = disabled" || exit 1

echo "running: CLICOLOR_FORCE=1 NO_COLOR=1 ./shamrock --help (expect failure)"
if run_with_profile env CLICOLOR_FORCE=1 NO_COLOR=1 ./shamrock --help; then
    echo "Error: expected non-zero exit when CLICOLOR_FORCE and NO_COLOR are both set" >&2
    exit 1
fi

echo "running: ./shamrock --color (expect \\x1b\\[36m in output)"
run_with_profile ./shamrock --color | grep -P "\x1b\[36m" || exit 1

echo "running: ./shamrock --nocolor (expect no \\x1b\\[36m in output)"
if run_with_profile ./shamrock --nocolor | grep -P "\x1b\[36m"; then
    echo "Error: expected no ANSI cyan escape sequence in output" >&2
    exit 1
fi


echo "::endgroup::"


#############################################
# Run the unittests for different world sizes
#############################################

# The commands ran below are similar to this one
#  mpirun --allow-run-as-root --bind-to socket:overload-allowed --oversubscribe \
#   -n 1 -x LLVM_PROFILE_FILE=utests_$(next_val).profraw ./shamrock_test --smi-full --sycl-cfg "${SYCLCFG}" --unittest --loglevel 0 : \
#   -n 1 -x LLVM_PROFILE_FILE=utests_$(next_val).profraw ./shamrock_test --smi-full --sycl-cfg "${SYCLCFG}" --unittest --loglevel 0 : \
#   -n 1 -x LLVM_PROFILE_FILE=utests_$(next_val).profraw ./shamrock_test --smi-full --sycl-cfg "${SYCLCFG}" --unittest --loglevel 0 : \
#   -n 1 -x LLVM_PROFILE_FILE=utests_$(next_val).profraw ./shamrock_test --smi-full --sycl-cfg "${SYCLCFG}" --unittest --loglevel 0

for world_size in 1 2 3 4; do
    echo "::group::Shamrock Unittests world_size = ${world_size}"

    cmd=(
        mpirun ${MPIARGS}
    )

    for ((rank=0; rank<world_size; rank++)); do
        profile_file="$(next_profile_file)"

        cmd+=(
            -n 1
            -x LLVM_PROFILE_FILE="${profile_file}" # Used for coverage profiling
            ./shamrock_test
            --smi-full
            --sycl-cfg "${SYCLCFG}"
            --unittest
            --loglevel 0
        )

        if (( rank < world_size - 1 )); then
            cmd+=( :)
        fi
    done

    echo "Running command: ${cmd[@]}"
    "${cmd[@]}"

    echo "::endgroup::"
done
