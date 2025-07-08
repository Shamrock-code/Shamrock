# SolverGraph Migration Guide

This guide explains how to migrate from the legacy module architecture to the modern `solvergraph` architecture.

## Overview

The solvergraph architecture provides a more structured and efficient way to manage data dependencies and memory in Shamrock. This migration represents a significant architectural improvement that:

- Provides better integration with the solver framework
- Improves memory management through implicit memory management & explicit deallocation
- Enables better visualization and debugging of data dependencies
- Supports more sophisticated data flow patterns

The complete migration from legacy modules to solvergraph involves multiple phases, including migrating data storage components (edges) and computation logic (nodes).

## Edge Migration

### Step 1: Identify the Component to Migrate

First, identify the `shambase::StorageComponent<T>` that needs to be migrated. Look for patterns like:

```cpp
// Old pattern
Component<SomeDataType> my_component;
```

### Step 2: Determine the Appropriate SolverGraph Edge Type

Choose the appropriate solvergraph edge type based on your data:

#### Basic Data Edges

- **`IDataEdgeNamed`**: For simple named data with no special requirements
- **`ScalarEdge<T>`**: For scalar values
- **`Indexes<T>`**: For index arrays

#### Field-Based Edges

- **`Field<T>`**: For distributed fields with automatic memory management
- **`FieldRefs<T>`**: For references to existing fields
- **`FieldSpan<T>`**: For field spans (performance-oriented access)

#### Custom Edges

If none of the existing edge types fit your needs, you'll need to create a custom edge.

### Step 3: Create the SolverGraph Edge Implementation

#### Option A: Use Existing Edge Types

If your data fits an existing edge type, use it directly:

```cpp
// For scalar values
std::shared_ptr<shamrock::solvergraph::ScalarEdge<T>> my_scalar;

// For index arrays
std::shared_ptr<shamrock::solvergraph::Indexes<T>> my_indexes;

// For distributed fields
std::shared_ptr<shamrock::solvergraph::Field<T>> my_field;
```

#### Option B: Create Custom Edge

If you need a custom edge, create a new class inheriting from `IDataEdgeNamed`:

```cpp
// Header file: include/yourmodel/solvergraph/YourCustomEdge.hpp
#pragma once

#include "shamrock/solvergraph/IDataEdgeNamed.hpp"
#include "your_data_type.hpp"

namespace yourmodel::solvergraph {

    class YourCustomEdge : public shamrock::solvergraph::IDataEdgeNamed {
    public:
        using IDataEdgeNamed::IDataEdgeNamed;

        // Your data members
        YourDataType data;

        // Implement memory management if needed
        inline virtual void free_alloc() {
            data = {}; // Clear your data
        }
    };

} // namespace yourmodel::solvergraph
```

### Step 4: Update Storage Declaration

Replace the old component declaration with the new solvergraph edge:

```cpp
// Old pattern
Component<YourDataType> my_component;

// New pattern
std::shared_ptr<yourmodel::solvergraph::YourCustomEdge> my_edge;
```

### Step 5: Update Initialization

Update the initialization code to create the solvergraph edge:

```cpp
// Old pattern
// Component was typically initialized implicitly or through storage

// New pattern
my_edge = std::make_shared<yourmodel::solvergraph::YourCustomEdge>(
    "edge_name", "\\text{LaTeX Symbol}"
);
```

### Step 6: Update Usage Patterns

Update how the data is accessed and used:

```cpp
// Old pattern
auto& data = my_component.get();

// New pattern
// Direct access to member
auto& data = shambase::get_check_ref(my_edge).data;
// or
// If you provide accessor methods
auto& data = shambase::get_check_ref(my_edge).get_data();
```

### Step 7: Handle Memory Management

Ensure proper memory management by implementing `free_alloc()` if needed:

```cpp
// In your custom edge implementation
inline virtual void free_alloc() {
    // Clear any allocated memory
    data.clear();
    // or
    data = {};
}
```

Replace instances of `.reset()` on the storage component by `shambase::get_check_ref(my_edge).free_alloc()`.

## Common Migration Patterns

### Pattern 1: Simple Scalar Storage

**Before:**
```cpp
Component<MyDataType> my_data;
```

**After:**
```cpp
std::shared_ptr<shamrock::solvergraph::ScalarEdge<MyDataType>> my_data;
```

### Pattern 2: Distributed Data

**Before:**
```cpp
Component<shambase::DistributedData<MyType>> distributed_data;
```

**After:**
```cpp
std::shared_ptr<MyCustomEdge> distributed_data;
```

Where `MyCustomEdge` inherits from `IDataEdgeNamed` and contains:
```cpp
shambase::DistributedData<MyType> data;
```

### Pattern 3: Field Data

**Before:**
```cpp
Component<shamrock::ComputeField<T>> field_data;
```

**After:**
```cpp
std::shared_ptr<shamrock::solvergraph::Field<T>> field_data;
```

## Edge Migration Checklist

- [ ] Identify the component to migrate
- [ ] Choose appropriate solvergraph edge type
- [ ] Create custom edge implementation if needed
- [ ] Update storage declaration
- [ ] Update initialization code (in init_solvergraph function)
- [ ] Update usage patterns throughout the codebase
- [ ] Replace `.reset()` by `.free_alloc()`
- [ ] Test the edge migration thoroughly
- [ ] Update any related documentation

## Testing Your Edge Migration

1. **Compilation**: Ensure the code compiles without errors
2. **Functionality**: Verify that the migrated edge works as expected
3. **Memory**: Check for memory leaks using appropriate tools
4. **Performance**: Ensure performance is maintained or improved
5. **Integration**: Test integration with other solvergraph edges

## Common Pitfalls

1. **Missing Memory Management**: Always implement `free_alloc()` if your edge manages memory
2. **Incorrect Edge Type**: Choose the most appropriate edge type for your data
3. **Incomplete Migration**: Ensure all usage patterns are updated
4. **Missing Dependencies**: Include all necessary headers for solvergraph components

## Example: Complete Edge Migration

Here's a complete example of migrating a neighbor cache component to a solvergraph edge:

**Before:**
```cpp
// In SolverStorage.hpp
Component<shamrock::tree::ObjectCacheHandler> neighbors_cache;

// In Solver.cpp
// Component was used implicitly
```

**After:**
```cpp
// In SolverStorage.hpp
std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache;

// In Solver.cpp
storage.neigh_cache = std::make_shared<shammodels::sph::solvergraph::NeighCache>(
    "neigh_cache", "neigh"
);
```

**Custom Edge Implementation:**
```cpp
// In NeighCache.hpp
class NeighCache : public shamrock::solvergraph::IDataEdgeNamed {
public:
    using IDataEdgeNamed::IDataEdgeNamed;

    shambase::DistributedData<shamrock::tree::ObjectCache> neigh_cache;

    shamrock::tree::ObjectCache &get_cache(u64 id) {
        return neigh_cache.get(id);
    }

    inline virtual void free_alloc() {
        neigh_cache = {};
    }
};
```

## Node Migration

TODO: This section will be documented in another PR.

## Conclusion

This migration process provides a structured approach to moving from the legacy module system to the modern solvergraph architecture. The benefits include better memory management, improved integration, and enhanced debugging capabilities.

For additional help or questions about specific migration scenarios, consult the existing solvergraph implementations in the codebase or refer to the solvergraph API documentation.
