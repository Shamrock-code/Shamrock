# Profiling build performance / time

Install [ClangBuildAnalyzer](https://github.com/aras-p/ClangBuildAnalyzer). On macOS: `brew install clang-build-analyzer`. Add `-ftime-trace` to CXX flags.

Then in the build dir:

```bash
shammake clean
ccache -C

ClangBuildAnalyzer --start .
shammake
ClangBuildAnalyzer --stop . capture_build.bin
ClangBuildAnalyzer --analyze capture_build.bin
```

## Example output (2026-08-04)

```
Analyzing build trace from 'capture_build.bin'...
**** Time summary:
Compilation (418 times):
  Parsing (frontend):         1335.9 s
  Codegen & opts (backend):   1322.7 s

**** Files that took longest to parse (compiler frontend):
 21505 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/pySPHModel.cpp.o
 11682 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Solver.cpp.o
 10291 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/ComputeEos.cpp.o
  9192 ms: ./src/shammodels/gsph/CMakeFiles/shammodels_gsph.dir/src/Solver.cpp.o
  8861 ms: ./src/shamrock/CMakeFiles/shamlib.dir/src/patch/PatchDataField.cpp.o
  8600 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Model.cpp.o
  8297 ms: ./src/shammodels/gsph/CMakeFiles/shammodels_gsph.dir/src/pyGSPHModel.cpp.o
  8283 ms: ./src/shamalgs/CMakeFiles/shamalgs.dir/src/details/reduction/reduction.cpp.o
  7404 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/render/CartesianRender.cpp.o
  7392 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/render/RenderFieldGetter.cpp.o
  7338 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/SPHSetup.cpp.o
  7254 ms: ./src/shammodels/ramses/CMakeFiles/shammodels_ramses.dir/src/Solver.cpp.o
  7183 ms: ./src/shamrock/CMakeFiles/shamlib.dir/src/patch/PatchDataLayer.cpp.o
  7119 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/UpdateDerivs.cpp.o
  7087 ms: ./src/shammodels/ramses/CMakeFiles/shammodels_ramses.dir/src/pyRamsesModel.cpp.o
  7045 ms: ./src/CMakeFiles/shamrock_test.dir/tests/shammodels/sph/comp_phantom_sedov.cpp.o
  6993 ms: ./src/shammodels/ramses/CMakeFiles/shammodels_ramses.dir/src/modules/NodeComputeFlux.cpp.o
  6886 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/AnalysisDisc.cpp.o
  6841 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/io/VTKDump.cpp.o
  6740 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/ExternalForces.cpp.o

**** Files that took longest to codegen (compiler backend):
 32901 ms: ./src/shamrock/CMakeFiles/shamlib.dir/src/patch/PatchDataField.cpp.o
 32321 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Solver.cpp.o
 28784 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/pySPHModel.cpp.o
 28613 ms: ./src/CMakeFiles/shamrock_test.dir/tests/shammath/sphkernelsTests.cpp.o
 26482 ms: ./src/shamalgs/CMakeFiles/shamalgs.dir/src/details/reduction/reduction.cpp.o
 23325 ms: ./src/shammodels/ramses/CMakeFiles/shammodels_ramses.dir/src/modules/InterpolateToFace.cpp.o
 22908 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/ComputeEos.cpp.o
 19963 ms: ./src/shamalgs/CMakeFiles/shamalgs.dir/src/details/reduction/groupReduction_usm.cpp.o
 16882 ms: ./src/shammodels/gsph/CMakeFiles/shammodels_gsph.dir/src/Solver.cpp.o
 16796 ms: ./src/shamtree/CMakeFiles/shamtree.dir/src/RadixTree.cpp.o
 15809 ms: ./src/shamrock/CMakeFiles/shamlib.dir/src/patch/PatchDataLayer.cpp.o
 13321 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/SPHSetup.cpp.o
 12979 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Model.cpp.o
 11676 ms: ./src/shamalgs/CMakeFiles/shamalgs.dir/src/details/algorithm/algorithm.cpp.o
 11620 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/UpdateDerivs.cpp.o
 11543 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/ExternalForces.cpp.o
 11288 ms: ./src/shammodels/common/CMakeFiles/shammodels_common.dir/src/pyCommonUtils.cpp.o
 11210 ms: ./src/shammodels/ramses/CMakeFiles/shammodels_ramses.dir/src/Solver.cpp.o
 10781 ms: ./src/shamsys/CMakeFiles/shamsys.dir/src/legacy/sycl_mpi_interop.cpp.o
 10695 ms: ./src/CMakeFiles/shamrock_test.dir/tests/fmmTests.cpp.o

**** Templates that took longest to instantiate:
 35029 ms: shamrock::patch::FieldVariant<shamrock::patch::PatchDataLayerLayout::FieldDescriptor>::visit<(lambda at /Users/davidcl... (2057 times, avg 17 ms)
 34889 ms: std::visit<(lambda at /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamrock/include/shamrock/patch/Field... (2059 times, avg 16 ms)
 34594 ms: std::__variant_detail::__visitation::__variant::__visit_value<(lambda at /Users/davidclt/Documents/shamrock-dev/Shamro... (2059 times, avg 16 ms)
 34170 ms: std::__variant_detail::__visitation::__variant::__visit_alt<std::__variant_detail::__visitation::__variant::__value_vi... (2059 times, avg 16 ms)
 33938 ms: std::__variant_detail::__visitation::__base::__visit_alt<std::__variant_detail::__visitation::__variant::__value_visit... (2059 times, avg 16 ms)
 33086 ms: std::__variant_detail::__visitation::__base::__make_fmatrix<std::__variant_detail::__visitation::__variant::__value_vi... (2059 times, avg 16 ms)
 32782 ms: std::__variant_detail::__visitation::__base::__make_fmatrix_impl<std::__variant_detail::__visitation::__variant::__val... (2059 times, avg 15 ms)
 29602 ms: std::visit<(lambda at /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamrock/include/shamrock/patch/Patch... (118 times, avg 250 ms)
 29566 ms: std::__variant_detail::__visitation::__variant::__visit_value<(lambda at /Users/davidclt/Documents/shamrock-dev/Shamro... (118 times, avg 250 ms)
 29549 ms: std::__variant_detail::__visitation::__variant::__visit_alt<std::__variant_detail::__visitation::__variant::__value_vi... (118 times, avg 250 ms)
 29527 ms: std::__variant_detail::__visitation::__base::__visit_alt<std::__variant_detail::__visitation::__variant::__value_visit... (118 times, avg 250 ms)
 29293 ms: std::__variant_detail::__visitation::__base::__make_fmatrix<std::__variant_detail::__visitation::__variant::__value_vi... (118 times, avg 248 ms)
 29268 ms: std::__variant_detail::__visitation::__base::__make_fmatrix_impl<std::__variant_detail::__visitation::__variant::__val... (118 times, avg 248 ms)
 20080 ms: nlohmann::basic_json<>::parse<const char *> (136 times, avg 147 ms)
 18184 ms: shamrock::patch::FieldVariant<PatchDataField>::visit<(lambda at /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor... (118 times, avg 154 ms)
 17864 ms: hipsycl::sycl::handler::parallel_for<__acpp_unnamed_kernel, (lambda at /Users/davidclt/Documents/shamrock-dev/Shamrock... (860 times, avg 20 ms)
 17599 ms: nlohmann::detail::parser<nlohmann::basic_json<>, nlohmann::detail::iterator_input_adapter<const char *>>::parse (136 times, avg 129 ms)
 14871 ms: hipsycl::sycl::detail::separate_last_argument_and_apply<(lambda at /opt/homebrew/Cellar/adaptivecpp/25.10.0_1/bin/../i... (860 times, avg 17 ms)
 13672 ms: nlohmann::detail::parser<nlohmann::basic_json<>, nlohmann::detail::iterator_input_adapter<const char *>>::sax_parse_in... (136 times, avg 100 ms)
 13035 ms: fmt::detail::write<char, fmt::basic_appender<char>, float, 0> (774 times, avg 16 ms)
  9833 ms: nlohmann::basic_json<>::basic_json (421 times, avg 23 ms)
  9676 ms: std::visit<(lambda at /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamrock/include/shamrock/patch/Field... (723 times, avg 13 ms)
  9595 ms: std::__variant_detail::__visitation::__variant::__visit_value<(lambda at /Users/davidclt/Documents/shamrock-dev/Shamro... (723 times, avg 13 ms)
  9489 ms: std::__variant_detail::__visitation::__variant::__visit_alt<std::__variant_detail::__visitation::__variant::__value_vi... (723 times, avg 13 ms)
  9409 ms: std::__variant_detail::__visitation::__base::__visit_alt<std::__variant_detail::__visitation::__variant::__value_visit... (723 times, avg 13 ms)
  9138 ms: std::__variant_detail::__visitation::__base::__make_fmatrix<std::__variant_detail::__visitation::__variant::__value_vi... (723 times, avg 12 ms)
  9045 ms: std::__variant_detail::__visitation::__base::__make_fmatrix_impl<std::__variant_detail::__visitation::__variant::__val... (723 times, avg 12 ms)
  8791 ms: hipsycl::rt::settings::get_configuration_or_default<hipsycl::rt::setting::visibility_mask, std::unordered_map<hipsycl:... (370 times, avg 23 ms)
  8786 ms: hipsycl::sycl::handler::parallel_for<__acpp_unnamed_kernel, (lambda at /Users/davidclt/Documents/shamrock-dev/Shamrock... (448 times, avg 19 ms)
  8490 ms: hipsycl::sycl::detail::select_devices<hipsycl::sycl::device_selector> (370 times, avg 22 ms)

**** Template sets that took longest to instantiate:
 94402 ms: std::vector<$>::emplace_back<$> (14030 times, avg 6 ms)
 94170 ms: std::__variant_detail::__visitation::__base::__visit_alt<$> (4479 times, avg 21 ms)
 92342 ms: std::__variant_detail::__visitation::__base::__make_fmatrix<$> (4479 times, avg 20 ms)
 91748 ms: std::__variant_detail::__visitation::__base::__make_fmatrix_impl<$> (4479 times, avg 20 ms)
 90068 ms: std::visit<$> (4056 times, avg 22 ms)
 89495 ms: std::__variant_detail::__visitation::__variant::__visit_value<$> (4056 times, avg 22 ms)
 88812 ms: std::__variant_detail::__visitation::__variant::__visit_alt<$> (4056 times, avg 21 ms)
 81125 ms: std::vector<$>::push_back (11275 times, avg 7 ms)
 72544 ms: std::vector<$>::__swap_out_circular_buffer (13475 times, avg 5 ms)
 70755 ms: std::__make_exception_guard<$> (17642 times, avg 4 ms)
 68967 ms: std::vector<$>::__emplace_back_slow_path<$> (10454 times, avg 6 ms)
 67555 ms: std::__exception_guard_exceptions<$>::~__exception_guard_exceptions (15954 times, avg 4 ms)
 66556 ms: std::__uninitialized_allocator_relocate<$> (12425 times, avg 5 ms)
 64170 ms: shamrock::patch::FieldVariant<$>::visit<$> (2905 times, avg 22 ms)
 59945 ms: std::reverse_iterator<$> (15499 times, avg 3 ms)
 56820 ms: hipsycl::sycl::handler::parallel_for<$> (2798 times, avg 20 ms)
 50455 ms: std::__variant_detail::__visitation::__base::__make_dispatch<$> (32198 times, avg 1 ms)
 48622 ms: hipsycl::sycl::detail::separate_last_argument_and_apply<$> (2798 times, avg 17 ms)
 48051 ms: std::vector<$> (28187 times, avg 1 ms)
 43913 ms: std::__variant_detail::__visitation::__base::__dispatcher<$>::__dispatch<$> (26614 times, avg 1 ms)
 40826 ms: std::vector<$>::__init_with_size<$> (8012 times, avg 5 ms)
 40048 ms: std::vector<$>::vector (8542 times, avg 4 ms)
 39328 ms: std::vector<$>::__construct_at_end<$> (9014 times, avg 4 ms)
 39020 ms: shamrock::patch::PatchDataLayerLayout::add_field<$> (2057 times, avg 18 ms)
 37289 ms: std::__uninitialized_allocator_copy<$> (8832 times, avg 4 ms)
 28850 ms: std::unordered_map<$> (5497 times, avg 5 ms)
 27439 ms: std::allocator_traits<$>::construct<$> (5844 times, avg 4 ms)
 25671 ms: std::__hash_table<$> (5734 times, avg 4 ms)
 24424 ms: sham::details::typed_index_kernel_call<$> (860 times, avg 28 ms)
 24393 ms: sham::details::typed_index_kernel_call_lambda<$> (888 times, avg 27 ms)

**** Functions that took longest to compile:
   985 ms: void test_karras_alg<unsigned int>() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamrock/tree/karrasTests.cpp)
   932 ms: void test_karras_alg<unsigned long long>() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamrock/tree/karrasTests.cpp)
   899 ms: test_func_91() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shambackends/SYCL_MPI_types_tests.cpp)
   571 ms: test_func_241() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamtree/KarrasRadixTreeAABBTests.cpp)
   570 ms: test_func_24() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamtree/KarrasRadixTreeAABBTests.cpp)
   471 ms: test_func_460()::$_0::operator()() const (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamalgs/primitives/reductionTests.cpp)
   334 ms: shammodels::basegodunov::Solver<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >, hipsycl... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shammodels/ramses/src/Solver.cpp)
   314 ms: test_func_16() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamalgs/primitives/upper_boundTests.cpp)
   303 ms: test_func_24() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamtree/KarrasRadixTreeFieldTests.cpp)
   281 ms: test_func_26() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamtree/MortonReducedSetTests.cpp)
   279 ms: test_func_157() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamtree/MortonReducedSetTests.cpp)
   248 ms: test_func_16() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamalgs/primitives/lower_boundTests.cpp)
   240 ms: test_func_29() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamrock/solvergraph/SolverGraph_tests.cpp)
   220 ms: validate_dtt_results(sham::DeviceBuffer<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >,... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamtree/DTTTesting_tests.cpp)
   200 ms: shammodels::sph::Solver<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >, shammath::M4>::... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shammodels/sph/src/Solver.cpp)
   199 ms: void test_smoothing_length_density_module<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> ... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammodels/sph/modules/IterateSmoothingLengthDensityTests.cpp)
   198 ms: test_func_21() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamrock/patch/PatchDataFieldSpanTests.cpp)
   188 ms: shammodels::sph::Solver<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >, shammath::M8>::... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shammodels/sph/src/Solver.cpp)
   184 ms: shammodels::sph::Solver<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >, shammath::M6>::... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shammodels/sph/src/Solver.cpp)
   181 ms: void validate_kernel_3d<shammath::SPHKernelGen<double, shammath::details::KernelDefTGauss3<double> > >(shammath::SPHKe... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   178 ms: void validate_kernel_3d<shammath::SPHKernelGen<double, shammath::details::KernelDefM5<double> > >(shammath::SPHKernelG... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   177 ms: test_func_21()::$_0::operator()() const (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamalgs/primitives/reductionTests.cpp)
   177 ms: shammodels::sph::Solver<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >, shammath::C4>::... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shammodels/sph/src/Solver.cpp)
   173 ms: shammodels::sph::Solver<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >, shammath::C6>::... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shammodels/sph/src/Solver.cpp)
   172 ms: void validate_kernel_3d<shammath::SPHKernelGen<float, shammath::details::KernelDefM7<float> > >(shammath::SPHKernelGen... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   170 ms: void validate_kernel_3d<shammath::SPHKernelGen<double, shammath::details::KernelDefM6<double> > >(shammath::SPHKernelG... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   169 ms: test_func_51() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamrock/solvergraph/SolverGraphSerializable_tests.cpp)
   169 ms: void validate_kernel_3d<shammath::SPHKernelGen<float, shammath::details::KernelDefM4Shift16<float> > >(shammath::SPHKe... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   168 ms: void validate_kernel_3d<shammath::SPHKernelGen<double, shammath::details::KernelDefM4Shift4<double> > >(shammath::SPHK... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   168 ms: void validate_kernel_3d<shammath::SPHKernelGen<float, shammath::details::KernelDefC2<float> > >(shammath::SPHKernelGen... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   166 ms: shammodels::sph::Solver<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >, shammath::C2>::... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shammodels/sph/src/Solver.cpp)
   166 ms: test_func_13() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shambase/FixedStackTests.cpp)
   165 ms: test_func_23()::$_0::operator()() const (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shamalgs/primitives/segmented_sort_in_placeTests.cpp)
   164 ms: void validate_kernel_3d<shammath::SPHKernelGen<float, shammath::details::KernelDefM4DoubleHump3<float> > >(shammath::S... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   163 ms: void validate_kernel_3d<shammath::SPHKernelGen<double, shammath::details::KernelDefC4<double> > >(shammath::SPHKernelG... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   161 ms: shampylib::init_shamrock_math_sphkernels(pybind11::module_&) (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shampylib/src/math/pySPHKernels.cpp)
   160 ms: void validate_kernel_3d<shammath::SPHKernelGen<float, shammath::details::KernelDefM9<float> > >(shammath::SPHKernelGen... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   159 ms: shammodels::sph::modules::SPHSetup<hipsycl::sycl::vec<double, 3, hipsycl::sycl::detail::vec_storage<double, 3> >, sham... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shammodels/sph/src/modules/SPHSetup.cpp)
   157 ms: void validate_kernel_3d<shammath::SPHKernelGen<double, shammath::details::KernelDefC6<double> > >(shammath::SPHKernelG... (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shammath/sphkernelsTests.cpp)
   153 ms: test_func_21() (/Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/tests/shambackends/kernel_call_distribTests.cpp)

**** Function sets that took longest to compile / optimize:
 12026 ms: fmt::v12::detail::format_dragon(fmt::v12::detail::basic_fp<$>, unsigned int, int, fmt::v12::detail::buffer<$>&, int&) (312 times, avg 38 ms)
  6275 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write_escaped_string<$>(fmt::v12::basic_appender<$>, fmt::v12::basic_str... (624 times, avg 10 ms)
  5905 ms: void validate_kernel_3d<$>(shammath::SPHKernelGen<$>::Tscal, shammath::SPHKernelGen<$>::Tscal, shammath::SPHKernelGen<... (40 times, avg 147 ms)
  4749 ms: hipsycl::sycl::event hipsycl::sycl::queue::submit<$>(hipsycl::sycl::property_list const&, hipsycl::sycl::vec<$>) (463 times, avg 10 ms)
  4705 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write_int_noinline<$>(fmt::v12::basic_appender<$>, fmt::v12::detail::wri... (936 times, avg 5 ms)
  4007 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write_fixed<$>(fmt::v12::basic_appender<$>, fmt::v12::detail::dragonbox:... (1246 times, avg 3 ms)
  2350 ms: hipsycl::sycl::accessor<$>::init_host_buffer(hipsycl::rt::runtime*, bool) (261 times, avg 9 ms)
  2226 ms: int fmt::v12::detail::format_float<$>(double, int, fmt::v12::format_specs const&, bool, fmt::v12::detail::buffer<$>&) (312 times, avg 7 ms)
  2172 ms: void fmt::v12::detail::parse_format_string<$>(fmt::v12::basic_string_view<$>, fmt::v12::detail::format_handler<$>&&) (311 times, avg 6 ms)
  1918 ms: void test_karras_alg<$>() (2 times, avg 959 ms)
  1762 ms: fmt::v12::basic_appender<$> fmt::v12::detail::do_write_float<$>(fmt::v12::basic_appender<$>, fmt::v12::detail::dragonb... (624 times, avg 2 ms)
  1706 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write_int<$>(fmt::v12::basic_appender<$>, unsigned __int128, unsigned in... (312 times, avg 5 ms)
  1683 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write_significand<$>(fmt::v12::basic_appender<$>, char const*, int, int,... (312 times, avg 5 ms)
  1606 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write_padded<$>(fmt::v12::basic_appender<$>, fmt::v12::format_specs cons... (1246 times, avg 1 ms)
  1597 ms: void fmt::v12::detail::format_hexfloat<$>(double, fmt::v12::format_specs, fmt::v12::detail::buffer<$>&) (312 times, avg 5 ms)
  1589 ms: PatchDataField<$>::get_obj_cnt() const (1293 times, avg 1 ms)
  1529 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write_padded<$>(fmt::v12::basic_appender<$>, fmt::v12::format_specs cons... (624 times, avg 2 ms)
  1501 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write<$>(fmt::v12::basic_appender<$>, fmt::v12::basic_string_view<$>, fm... (936 times, avg 1 ms)
  1479 ms: void test_comm<$>() (60 times, avg 24 ms)
  1448 ms: fmt::v12::detail::format_handler<$>::on_format_specs(int, char const*, char const*) (311 times, avg 4 ms)
  1443 ms: pybind11::cpp_function::initialize_generic(std::__1::unique_ptr<$>&&, char const*, std::type_info const* const*, unsig... (39 times, avg 37 ms)
  1429 ms: decltype(fp0.out()) fmt::v12::range_formatter<$>::format<$>(std::__1::vector<$> const&, fmt::v12::context&) const (239 times, avg 5 ms)
  1372 ms: char const* fmt::v12::detail::parse_format_specs<$>(char const*, char const*, fmt::v12::detail::dynamic_format_specs<$... (311 times, avg 4 ms)
  1338 ms: fmt::v12::basic_appender<$> fmt::v12::detail::digit_grouping<$>::apply<$>(fmt::v12::basic_appender<$>, fmt::v12::basic... (312 times, avg 4 ms)
  1264 ms: std::__1::shared_ptr<$> hipsycl::sycl::queue::execute_submission<$>(hipsycl::sycl::vec<$>, hipsycl::sycl::handler&) (463 times, avg 2 ms)
  1255 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write<$>(fmt::v12::basic_appender<$>, float) (311 times, avg 4 ms)
  1246 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write_fixed<$>(fmt::v12::basic_appender<$>, fmt::v12::detail::big_decima... (312 times, avg 3 ms)
  1223 ms: hipsycl::sycl::event hipsycl::sycl::queue::submit<$>(hipsycl::sycl::property_list const&, double) (123 times, avg 9 ms)
  1216 ms: void sham::kernel_call<$>(sham::DeviceQueue&, auto, auto, unsigned int, auto&&, SourceLocation&&) (30 times, avg 40 ms)
  1127 ms: fmt::v12::basic_appender<$> fmt::v12::detail::write_int<$>(fmt::v12::basic_appender<$>, unsigned long long, unsigned i... (312 times, avg 3 ms)

**** Expensive headers:
393716 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambackends/include/shambackends/sycl.hpp (included 369 times, avg 1066 ms), included via:
  31x: shamtest.hpp Test.hpp TestResult.hpp TestAssertList.hpp
  28x: DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  17x: reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  11x: math.hpp
  9x: random.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  8x: NodeInstance.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  8x: random.hpp random.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  7x: AABB.hpp math.hpp
  7x: kernel_call_distrib.hpp kernel_call.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  7x: kernel_call.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  6x: sphkernels.hpp math.hpp
  5x: Patch.hpp PatchCoord.hpp
  5x: <direct include>
  5x: memory.hpp memory.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  4x: bitonicSort.hpp
  4x: hilbert.hpp
  4x: CommunicationBuffer.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  4x: numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  4x: mock_vector.hpp random.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp Device.hpp
  3x: patchdata.hpp sycl_vector_utils.hpp
  3x: SolverConfig.hpp math.hpp
  ...

215990 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambackends/include/shambackends/DeviceBuffer.hpp (included 266 times, avg 811 ms), included via:
  29x: <direct include>
  17x: reduction.hpp flatten.hpp
  10x: random.hpp
  8x: random.hpp random.hpp
  7x: kernel_call.hpp
  7x: kernel_call_distrib.hpp kernel_call.hpp
  7x: memory.hpp memory.hpp
  4x: CommunicationBuffer.hpp
  4x: numeric.hpp reduction.hpp flatten.hpp
  4x: algorithm.hpp algorithm.hpp sort_by_keys.hpp
  4x: mock_vector.hpp random.hpp
  4x: PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp
  4x: patchdata.hpp PatchDataLayer.hpp PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp
  4x: PatchDataLayer.hpp PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp
  3x: reduction.hpp
  3x: is_all_true.hpp
  3x: numericTests.hpp random.hpp
  3x: serialize.hpp
  3x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp distributedDataComm.hpp sparseXchg.hpp reduction.hpp flatten.hpp
  3x: CompressedLeafBVH.hpp CLBVHObjectIterator.hpp
  2x: sparse_exchange.hpp
  ...

202414 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambackends/include/shambackends/Device.hpp (included 210 times, avg 963 ms), included via:
  28x: DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  17x: reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  9x: random.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  9x: NodeInstance.hpp DeviceScheduler.hpp DeviceQueue.hpp
  8x: random.hpp random.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  7x: kernel_call_distrib.hpp kernel_call.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  7x: kernel_call.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  5x: memory.hpp memory.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  4x: CommunicationBuffer.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  4x: numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  4x: mock_vector.hpp random.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  3x: algorithm.hpp algorithm.hpp sort_by_keys.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  3x: is_all_true.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  3x: reduction.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  3x: PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  3x: numericTests.hpp random.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  2x: <direct include>
  2x: sparse_exchange.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  2x: USMPtrHolder.hpp DeviceScheduler.hpp DeviceQueue.hpp
  2x: algorithm.hpp sort_by_keys.hpp DeviceBuffer.hpp DeviceScheduler.hpp DeviceQueue.hpp
  2x: EventList.hpp DeviceContext.hpp
  ...

202391 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambackends/include/shambackends/DeviceQueue.hpp (included 288 times, avg 702 ms), included via:
  28x: DeviceBuffer.hpp DeviceScheduler.hpp
  17x: reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  16x: NodeInstance.hpp DeviceScheduler.hpp
  9x: random.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  8x: random.hpp random.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  7x: kernel_call_distrib.hpp kernel_call.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  7x: kernel_call.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  7x: memory.hpp memory.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  4x: CommunicationBuffer.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  4x: numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  4x: mock_vector.hpp random.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  4x: algorithm.hpp algorithm.hpp sort_by_keys.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  4x: PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  4x: patchdata.hpp PatchDataLayer.hpp PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  4x: PatchDataLayer.hpp PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  3x: is_all_true.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  3x: reduction.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  3x: numericTests.hpp random.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  3x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp distributedDataComm.hpp sparseXchg.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  3x: serialize.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  3x: CompressedLeafBVH.hpp CLBVHObjectIterator.hpp DeviceBuffer.hpp DeviceScheduler.hpp
  ...

201278 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambackends/include/shambackends/DeviceScheduler.hpp (included 287 times, avg 701 ms), included via:
  28x: DeviceBuffer.hpp
  17x: reduction.hpp flatten.hpp DeviceBuffer.hpp
  16x: NodeInstance.hpp
  9x: random.hpp DeviceBuffer.hpp
  8x: random.hpp random.hpp DeviceBuffer.hpp
  7x: kernel_call_distrib.hpp kernel_call.hpp DeviceBuffer.hpp
  7x: kernel_call.hpp DeviceBuffer.hpp
  7x: memory.hpp memory.hpp DeviceBuffer.hpp
  4x: CommunicationBuffer.hpp DeviceBuffer.hpp
  4x: numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp
  4x: mock_vector.hpp random.hpp DeviceBuffer.hpp
  4x: algorithm.hpp algorithm.hpp sort_by_keys.hpp DeviceBuffer.hpp
  4x: PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp
  4x: patchdata.hpp PatchDataLayer.hpp PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp
  4x: PatchDataLayer.hpp PatchDataField.hpp numeric.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp
  3x: is_all_true.hpp DeviceBuffer.hpp
  3x: reduction.hpp DeviceBuffer.hpp
  3x: numericTests.hpp random.hpp DeviceBuffer.hpp
  3x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp distributedDataComm.hpp sparseXchg.hpp reduction.hpp flatten.hpp DeviceBuffer.hpp
  3x: serialize.hpp DeviceBuffer.hpp
  3x: CompressedLeafBVH.hpp CLBVHObjectIterator.hpp DeviceBuffer.hpp
  ...

97424 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambase/include/shambase/string.hpp (included 385 times, avg 253 ms), included via:
  43x: stacktrace.hpp
  28x: shamtest.hpp Test.hpp TestResult.hpp TestAssertList.hpp
  26x: <direct include>
  24x: memory.hpp
  18x: DeviceBuffer.hpp memory.hpp
  17x: time.hpp
  17x: logs.hpp logs.hpp msgformat.hpp
  16x: DistributedData.hpp
  7x: random.hpp random.hpp DeviceBuffer.hpp memory.hpp
  5x: TestResult.hpp TestAssertList.hpp
  5x: indexing.hpp logs.hpp logs.hpp msgformat.hpp
  4x: narrowing.hpp stacktrace.hpp
  3x: PatchDataLayer.hpp memory.hpp
  3x: memory.hpp memory.hpp
  3x: numeric.hpp memory.hpp
  3x: reduction.hpp flatten.hpp DeviceBuffer.hpp memory.hpp
  3x: NodeInstance.hpp DeviceScheduler.hpp DeviceQueue.hpp memory.hpp
  3x: serialize.hpp memory.hpp
  3x: algorithm.hpp algorithm.hpp sort_by_keys.hpp DeviceBuffer.hpp memory.hpp
  3x: reduction.hpp DeviceBuffer.hpp memory.hpp
  3x: sycl_utils.hpp stacktrace.hpp
  ...

87619 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamrock/include/shamrock/scheduler/PatchScheduler.hpp (included 101 times, avg 867 ms), included via:
  5x: Field.hpp ComputeField.hpp
  3x: AMRGrid.hpp
  3x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp
  3x: Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp
  3x: Model.hpp Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp
  3x: <direct include>
  3x: SPHUtilities.hpp BasicSPHGhosts.hpp ComputeField.hpp
  3x: FindGhostLayerCandidates.hpp SerialPatchTreeEdge.hpp SerialPatchTree.hpp
  2x: CopyPatchDataField.hpp Field.hpp ComputeField.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp SolverStorage.hpp ComputeField.hpp
  2x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp
  2x: ComputeLoadBalanceValue.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp
  2x: ResidualDot.hpp SolverConfig.hpp
  2x: BasicSPHGhosts.hpp ComputeField.hpp
  1x: ShamrockDump.hpp
  1x: SerialPatchTree.hpp
  1x: ExtractGhostField.hpp CopyPatchDataField.hpp Field.hpp ComputeField.hpp
  1x: AMROverheadtest.hpp AMRGrid.hpp
  1x: GhostZones.hpp Solver.hpp SolverConfig.hpp SolverStorage.hpp ComputeField.hpp
  ...

72758 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamrock/include/shamrock/patch/PatchDataLayer.hpp (included 118 times, avg 616 ms), included via:
  7x: <direct include>
  5x: patchdata.hpp
  3x: FindGhostLayerCandidates.hpp PatchDataLayerDDShared.hpp
  3x: Field.hpp ComputeField.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  3x: AMRGrid.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  3x: SPHUtilities.hpp BasicSPHGhosts.hpp
  3x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  3x: Model.hpp Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  3x: Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  2x: KillParticles.hpp PatchDataLayerRefs.hpp
  2x: PatchDataToPy.hpp
  2x: BasicSPHGhosts.hpp
  2x: CopyPatchDataField.hpp Field.hpp ComputeField.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp SolverStorage.hpp ComputeField.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  2x: ComputeLoadBalanceValue.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  2x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  2x: ResidualDot.hpp SolverConfig.hpp PatchScheduler.hpp PatchDataLayerRefs.hpp
  1x: CopyPatchDataLayerFields.hpp
  1x: FuseGhostLayer.hpp IPatchDataLayerRefs.hpp
  ...

62254 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambackends/include/shambackends/math.hpp (included 238 times, avg 261 ms), included via:
  30x: <direct include>
  21x: AABB.hpp
  17x: sphkernels.hpp
  6x: CoordRange.hpp intervals.hpp
  6x: Patch.hpp PatchCoord.hpp CoordRange.hpp intervals.hpp
  5x: morton.hpp
  5x: PatchDataLayer.hpp Patch.hpp PatchCoord.hpp CoordRange.hpp intervals.hpp
  4x: equals.hpp
  3x: SolverConfig.hpp
  3x: riemann.hpp
  3x: Solver.hpp SolverConfig.hpp
  3x: patchdata.hpp PatchDataLayer.hpp Patch.hpp PatchCoord.hpp CoordRange.hpp intervals.hpp
  3x: crystalLattice.hpp CoordRange.hpp intervals.hpp
  3x: SPHUtilities.hpp BasicSPHGhosts.hpp PatchDataLayer.hpp Patch.hpp PatchCoord.hpp CoordRange.hpp intervals.hpp
  3x: Field.hpp ComputeField.hpp PatchScheduler.hpp distributedDataComm.hpp sparseXchg.hpp
  2x: AMRGrid.hpp AMRCell.hpp
  2x: CompressedLeafBVH.hpp morton.hpp
  2x: ComputeLoadBalanceValue.hpp SolverConfig.hpp
  2x: TreeMortonCodes.hpp CoordRange.hpp intervals.hpp
  2x: distributedDataComm.hpp sparseXchg.hpp
  2x: Model.hpp Solver.hpp AMRBlock.hpp AABB.hpp
  ...

56953 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambase/include/shambase/SourceLocation.hpp (included 332 times, avg 171 ms), included via:
  52x: exception.hpp
  27x: stacktrace.hpp
  18x: shamtest.hpp Test.hpp TestResult.hpp TestAssertList.hpp
  16x: DistributedData.hpp exception.hpp
  16x: time.hpp string.hpp exception.hpp
  15x: DeviceBuffer.hpp memory.hpp exception.hpp
  13x: memory.hpp exception.hpp
  12x: string.hpp exception.hpp
  10x: logs.hpp logs.hpp msgformat.hpp string.hpp exception.hpp
  4x: <direct include>
  4x: TestResult.hpp TestAssertList.hpp
  4x: AABB.hpp
  4x: indexing.hpp logs.hpp logs.hpp msgformat.hpp string.hpp exception.hpp
  3x: PatchDataLayer.hpp exception.hpp
  3x: narrowing.hpp
  3x: SolverConfig.hpp exception.hpp
  3x: profiling.hpp
  3x: patchdata.hpp PatchDataLayer.hpp exception.hpp
  3x: morton.hpp CoordRangeTransform.hpp CoordRange.hpp
  2x: PatchDataLayerLayout.hpp
  2x: NodeInstance.hpp DeviceScheduler.hpp DeviceQueue.hpp memory.hpp exception.hpp
  ...

52950 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamalgs/include/shamalgs/collective/reduction.hpp (included 181 times, avg 292 ms), included via:
  17x: <direct include>
  15x: PatchDataField.hpp numeric.hpp
  7x: PatchDataLayer.hpp PatchDataField.hpp numeric.hpp
  6x: numeric.hpp
  5x: numeric.hpp numeric.hpp
  5x: patchdata.hpp PatchDataLayer.hpp PatchDataField.hpp numeric.hpp
  3x: SPHUtilities.hpp BasicSPHGhosts.hpp PatchDataField.hpp numeric.hpp
  3x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp distributedDataComm.hpp sparseXchg.hpp
  3x: Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp distributedDataComm.hpp sparseXchg.hpp
  3x: AMRBlock.hpp TreeTraversal.hpp numeric.hpp numeric.hpp
  3x: AMRGrid.hpp AMRCell.hpp AMRBlock.hpp TreeTraversal.hpp numeric.hpp numeric.hpp
  3x: Model.hpp Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp distributedDataComm.hpp sparseXchg.hpp
  2x: sparseXchg.hpp
  2x: distributedDataComm.hpp sparseXchg.hpp
  2x: Model.hpp Solver.hpp AMRBlock.hpp TreeTraversal.hpp numeric.hpp numeric.hpp
  2x: Model.hpp Solver.hpp AMRBlock.hpp TreeTraversal.hpp numeric.hpp numeric.hpp
  2x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp distributedDataComm.hpp sparseXchg.hpp
  2x: ComputeLoadBalanceValue.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp distributedDataComm.hpp sparseXchg.hpp
  2x: PatchDataToPy.hpp PatchDataLayer.hpp PatchDataField.hpp numeric.hpp
  2x: ResidualDot.hpp SolverConfig.hpp AMRBlock.hpp TreeTraversal.hpp numeric.hpp numeric.hpp
  2x: Field.hpp ComputeField.hpp PatchDataField.hpp numeric.hpp
  ...

51267 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shammodels/sph/include/shammodels/sph/SolverConfig.hpp (included 34 times, avg 1507 ms), included via:
  4x: Solver.hpp
  3x: Model.hpp Solver.hpp
  2x: <direct include>
  2x: ComputeLoadBalanceValue.hpp
  2x: NeighbourCache.hpp
  1x: VTKDump.hpp
  1x: ConservativeCheck.hpp
  1x: ParticleReordering.hpp
  1x: ModifierOffset.hpp
  1x: BuildTrees.hpp
  1x: AnalysisSodTube.hpp
  1x: UpdateViscosity.hpp
  1x: DiffOperator.hpp
  1x: DiffOperatorDtDivv.hpp
  1x: CartesianRender.hpp
  1x: ModifierSplitPart.hpp
  1x: AnalysisBarycenter.hpp Model.hpp Solver.hpp
  1x: AnalysisDisc.hpp
  1x: GeneratorMCDisc.hpp
  1x: ExternalForces.hpp
  1x: RenderFieldGetter.hpp
  ...

51065 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamtest/details/TestResult.hpp (included 143 times, avg 357 ms), included via:
  104x: shamtest.hpp Test.hpp
  33x: <direct include>
  3x: numericTests.hpp shamtest.hpp Test.hpp
  2x: sortTests.hpp shamtest.hpp Test.hpp
  1x: Test.hpp

49506 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamtest/details/TestAssertList.hpp (included 144 times, avg 343 ms), included via:
  104x: shamtest.hpp Test.hpp TestResult.hpp
  33x: TestResult.hpp
  3x: numericTests.hpp shamtest.hpp Test.hpp TestResult.hpp
  2x: sortTests.hpp shamtest.hpp Test.hpp TestResult.hpp
  1x: Test.hpp TestResult.hpp
  1x: <direct include>

46339 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamrock/include/shamrock/solvergraph/PatchDataLayerRefs.hpp (included 108 times, avg 429 ms), included via:
  4x: Field.hpp ComputeField.hpp PatchScheduler.hpp
  3x: AMRGrid.hpp PatchScheduler.hpp
  3x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp
  3x: Model.hpp Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp
  3x: Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp
  3x: PatchScheduler.hpp
  3x: FindGhostLayerCandidates.hpp
  3x: SPHUtilities.hpp BasicSPHGhosts.hpp ComputeField.hpp PatchScheduler.hpp
  2x: KillParticles.hpp
  2x: CopyPatchDataField.hpp Field.hpp ComputeField.hpp PatchScheduler.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp SolverStorage.hpp ComputeField.hpp PatchScheduler.hpp
  2x: ComputeLoadBalanceValue.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp
  2x: SolverConfig.hpp units_json.hpp SerialPatchTree.hpp PatchScheduler.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp PatchScheduler.hpp
  2x: ResidualDot.hpp SolverConfig.hpp PatchScheduler.hpp
  2x: BasicSPHGhosts.hpp ComputeField.hpp PatchScheduler.hpp
  2x: CopyPatchDataLayerFields.hpp PatchDataLayerEdge.hpp
  1x: SerialPatchTree.hpp PatchScheduler.hpp
  1x: ShamrockDump.hpp PatchScheduler.hpp
  1x: AMROverheadtest.hpp AMRGrid.hpp PatchScheduler.hpp
  ...

45451 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamtest/details/Test.hpp (included 110 times, avg 413 ms), included via:
  104x: shamtest.hpp
  3x: numericTests.hpp shamtest.hpp
  2x: sortTests.hpp shamtest.hpp
  1x: <direct include>

45183 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamtest/shamtest.hpp (included 140 times, avg 322 ms), included via:
  135x: <direct include>
  3x: numericTests.hpp
  2x: sortTests.hpp

43349 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambase/include/shambase/exception.hpp (included 377 times, avg 114 ms), included via:
  56x: <direct include>
  31x: stacktrace.hpp string.hpp
  18x: shamtest.hpp Test.hpp TestResult.hpp TestAssertList.hpp string.hpp
  17x: string.hpp
  16x: DistributedData.hpp
  16x: time.hpp string.hpp
  15x: DeviceBuffer.hpp memory.hpp
  14x: logs.hpp logs.hpp msgformat.hpp string.hpp
  13x: memory.hpp
  7x: random.hpp random.hpp DeviceBuffer.hpp memory.hpp
  5x: indexing.hpp logs.hpp logs.hpp msgformat.hpp string.hpp
  3x: PatchDataLayer.hpp
  3x: SolverConfig.hpp
  3x: NodeInstance.hpp DeviceScheduler.hpp DeviceQueue.hpp memory.hpp
  3x: serialize.hpp
  3x: narrowing.hpp
  3x: reduction.hpp DeviceBuffer.hpp memory.hpp
  3x: TestResult.hpp TestAssertList.hpp string.hpp
  3x: patchdata.hpp PatchDataLayer.hpp
  2x: numeric.hpp memory.hpp
  2x: sparse_exchange.hpp DeviceBuffer.hpp memory.hpp
  ...

42839 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shambase/include/shambase/memory.hpp (included 151 times, avg 283 ms), included via:
  24x: <direct include>
  19x: DeviceBuffer.hpp
  8x: random.hpp random.hpp DeviceBuffer.hpp
  7x: NodeInstance.hpp DeviceScheduler.hpp DeviceQueue.hpp
  5x: random.hpp DeviceBuffer.hpp
  4x: patchdata.hpp PatchDataLayer.hpp
  3x: numeric.hpp
  3x: PatchDataLayer.hpp
  3x: reduction.hpp flatten.hpp DeviceBuffer.hpp
  3x: serialize.hpp
  3x: algorithm.hpp algorithm.hpp sort_by_keys.hpp DeviceBuffer.hpp
  3x: CommunicationBuffer.hpp DeviceBuffer.hpp
  3x: reduction.hpp DeviceBuffer.hpp
  2x: sparse_exchange.hpp DeviceBuffer.hpp
  2x: reduction.hpp
  2x: bitonicSort_updated_usm.hpp DeviceBuffer.hpp
  2x: kernel_call.hpp DeviceBuffer.hpp
  2x: is_all_true.hpp DeviceBuffer.hpp
  2x: numericTests.hpp random.hpp DeviceBuffer.hpp
  1x: ErrorChecker.hpp DeviceBuffer.hpp
  1x: numericFallback.hpp DeviceBuffer.hpp
  ...

40761 ms: /Users/davidclt/Documents/shamrock-dev/Shamrock_cursor/src/shamrock/include/shamrock/scheduler/SerialPatchTree.hpp (included 74 times, avg 550 ms), included via:
  3x: SolverConfig.hpp units_json.hpp
  3x: Solver.hpp SolverConfig.hpp units_json.hpp
  3x: Model.hpp Solver.hpp SolverConfig.hpp units_json.hpp
  3x: FindGhostLayerCandidates.hpp SerialPatchTreeEdge.hpp
  3x: SPHUtilities.hpp BasicSPHGhosts.hpp InterfacesUtility.hpp
  2x: SolverConfig.hpp units_json.hpp
  2x: ComputeLoadBalanceValue.hpp SolverConfig.hpp units_json.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp units_json.hpp
  2x: Model.hpp Solver.hpp SolverConfig.hpp SolverStorage.hpp InterfacesUtility.hpp
  2x: Model.hpp Solver.hpp SolverStorage.hpp FindGhostLayerCandidates.hpp SerialPatchTreeEdge.hpp
  2x: BasicSPHGhosts.hpp InterfacesUtility.hpp
  1x: <direct include>
  1x: UpdateViscosity.hpp SolverConfig.hpp units_json.hpp
  1x: BuildTrees.hpp SolverConfig.hpp units_json.hpp
  1x: UpdateDerivs.hpp SolverConfig.hpp units_json.hpp
  1x: DiffOperatorDtDivv.hpp SolverConfig.hpp units_json.hpp
  1x: ConservativeCheck.hpp SolverConfig.hpp units_json.hpp
  1x: AnalysisSodTube.hpp SolverConfig.hpp units_json.hpp
  1x: DiffOperator.hpp SolverConfig.hpp units_json.hpp
  1x: ModifierOffset.hpp SolverConfig.hpp units_json.hpp
  1x: VTKDump.hpp SolverConfig.hpp units_json.hpp
  ...

  done in 0.5s.

```
