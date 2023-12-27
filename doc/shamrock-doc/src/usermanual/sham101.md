# Shamrock 101: basics of the code

Everything you'll want to touch is in the src folder. Quite intuitively Shammodels hosts all of the models implemented in Shamrock. 
Lets consider the SPH model. All that is contained in the cpp file will be compiled with Shamrock. Hence only essential parts of the alogrithm (functions called many many times) go there, and you probably don't need to go in there.

Some functions are backend depended: for instance, how the min between two integers is computed using different algorithm on cuda or openCL. In order to not be bothered by that while coding, Shamrock uses SYCL. All the functions that are backend dependent go in .... namespace. This namespace also contains an alias to base functions that are independent of the backend. This is because vectorizing bbase functions depends on the backend?

Shamrock uses templates. In cpp, fnctions depend on the type of the arguments.Templates are used to avoid coding several times the same functions for different arguments. Here is the basic definition of a templated function:
template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ConservativeCheck<Tvec, SPHKernel>::check_conservation(
    Tscal gpart_mass) { 
    ...
    }

the templated function check_conservation takes 2 arguments, one of some type defined in the class Tvec, and another of some type defined in the class SPHKernel. Note that the class SPHKernel is itself templated, hence the template<class> keyword, while Tvec is not a templated class.
shammodels::sph::modules::ConservativeCheck are abstrations designed to "hide" some parts of the code the user does not need to know.

after templating a function, it is necessary to instanciate it at the end of the file.


ctrl + click on error to see the line in the file that fucked up
