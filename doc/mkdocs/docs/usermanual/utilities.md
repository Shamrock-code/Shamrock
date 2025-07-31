# Shamrock project utilities

## Code Checks

### License check

Every c++ file in shamrock must start with :
```c++
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//
```

if not the `buildbot/check_licencing.py` will tell you which file do not have the license banner.

### Pragma once check


In c++ when we include a header this header might include headers that were already included
by another one. In order to prevent double definitions either you can wrap the
header with include guards:
```c++
#ifndef BLABLA_H
#define BLABLA_H

...content of the header...

#endif
```
Or we can just add in the begining of the header before anything else:
```c++
#pragma once

...content of the header...
```

Both options do the same thing, however `#pragma once` is slightly faster and more conveninent.
Therefor in shamrock every header starts with the license banner followed by `#pragma once`

in order to check the correctness of pragma onces there is the `buildbot/check_pragma_once.py`
utility will tell you which file do not have the pragma once.

In the end every header starts with :

```c++
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

...content of the header...
```

### Doxygen file header

For the doxygen every c++ file has to include the following comment block :

```c++
/**
 * @file filename.cpp/hpp
 * @author authorname (authormail@someemail.fr)
 * @brief ...description...
 *
 */
```

This is checked by the `buildbot/check_doxygen_fileheader.py`
utility, additionally if the utility detect a wrong filename,
it **automatically replace** it by the correct one, hence you can just write a random filename,
which will can get automatically corrected by the script.
