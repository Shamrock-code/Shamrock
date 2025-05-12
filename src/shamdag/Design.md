# Dag graph design

## Data stealing

for exemple if a field only add some data to another one we may want to not allocate a new one. In that case a node can steal the data of an input link to pass it to the output link.

The process involves using `IDataEdge::report_data_stealing()` to report the the edge that data was stolen which in turn call `INode::report_data_stealing()` which just set the current node to unevaluated. Should it reset the node local data too ?
