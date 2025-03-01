# Sink particles

Sink particles in Shamrock's SPH model are implemented as a globally synchronized list of particles. They are also integrated using the SPH leapfrog.

## Usage

### Adding a new sink particle

A new sink particle can be added like so:
```py
import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

< ... sph setup ... >

# add a new sink:
# parameters: (mass, position, velocity, accretion radius)
model.add_sink(1,(0,0,0),(0,0,0),0.1)
```

### Recovering sinks data

Sink data can be queried using the function `get_sinks`, which return a list of dictionary containing the sink data. For example one can do the following:
```py
sinks = model.get_sinks()

x1,y1,z1 = sinks[0]["pos"]
x2,y2,z2 = sinks[1]["pos"]
```

For a more involved example to plot results with sinks data see [On the fly plots in SPH](../../usermanual/plotting.md).

## Internal API

### Storage of the sinks data

Sinks data is stored within the SPH solver storage `shammodels::sph::SolverStorage` in the member field explicitly called `sinks`.

For now they are stored as `std::vector<SinkParticle<Tvec>>`, where the field of the sinks are hardcoded in the struct `SinkParticle`. Its current definition is:
```c++
template<class Tvec>
struct SinkParticle {
    public:
    using Tscal = shambase::VecComponent<Tvec>;

    Tvec pos;
    Tvec velocity;
    Tvec sph_acceleration;
    Tvec ext_acceleration;
    Tscal mass;
    Tvec angular_momentum;
    Tscal accretion_radius;
};
```
There are plans to extend its definition to allow extension to more complex physics (MHD for example) however the exact way to do so is still unclear.

### Algorithmics

TODO
