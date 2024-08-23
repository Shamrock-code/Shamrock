# Shamrock Log system

The log library is within the communication library since those are non trivially coupled.

First you can start by including the logs lib :
```c++
#include "shamcomm/logs.hpp"
```

everything can be found either in, `shamcomm::logs` or `logger`. The latter is only a alias to the first one for convenience.

From that point the utilisation is quite simple . In every case the name of any log function is : `logger::<levelname>(_ln)`, where `levelname` is the log level name, and if you append the suffix `_ln` it will add a line return at the end of the print.

currently available ones are :

| Loglevel name  | loglevel value    | Description         |
| :---------: |:---------:| :----------------------------------: |
| `debug_alloc`       | 127 |  debug print for allocation   |
| `debug_mpi`       | 100 |  debug print for mpi  |
| `debug_sycl`       | 11 |  debug print for sycl  |
| `debug`       | 10 |  generic debug print  |
| `info`       | 1 |  print usefull optional information  |
| `normal`       | 0 |  normal logging level  |
| `warn`       | -1 |  warnings  |
| `err`       | -10 |  errors  |
