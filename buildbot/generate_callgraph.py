import os
import subprocess
from pathlib import Path

for path in Path(".").rglob("*.o"):
    print("Found :", path.name)
    break

    os.system(
        "../../../sycl_compilers/dpcpp/bin/clang-offload-bundler --input "
        + str(path.absolute())
        + " --type ll --output "
        + str(path.absolute())
        + ".llvm.off"
        + ' --targets "host-x86_64-unknown-linux-gnu" --unbundle'
    )
    os.system(
        "../../../sycl_compilers/dpcpp/bin/clang-offload-bundler --input "
        + str(path.absolute())
        + " --type ll --output "
        + str(path.absolute())
        + ".llvm.offspr"
        + ' --targets "sycl-spir64-unknown-unknown" --unbundle'
    )

    os.system(
        "../../../sycl_compilers/dpcpp/bin/opt -p dot-callgraph -o "
        + str(path.absolute())
        + ".tmp "
        + str(path.absolute())
        + ".llvm.off"
    )
    os.system(
        "../../../sycl_compilers/dpcpp/bin/opt -p dot-callgraph -o "
        + str(path.absolute())
        + ".tmp "
        + str(path.absolute())
        + ".llvm.offspr"
    )

for path in Path(".").rglob("*.callgraph.dot"):
    print("Found :", path.name)
    print("gen :", str(path.absolute()) + ".filt")

    os.system("cat " + str(path.absolute()) + " | c++filt > " + str(path.absolute()) + ".filt")


"""
old way
    with open(path.name+".callgraph.dot", 'w') as fout:
        p1 = subprocess.Popen(["cat", "<stdin>.callgraph.dot"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["c++filt"], stdin=p1.stdout, stdout=subprocess.PIPE)
        p3 = subprocess.Popen(["sed",'s,>,\\>,g; s,-\\>,->,g; s,<,\\<,g'], stdin=p2.stdout, stdout=fout)
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
        p2.stdout.close()
        output,err = p3.communicate()

    os.system("rm \<stdin\>.callgraph.dot")

"""

import graphviz

excludelist = [
    "std::",
    "sycl::",
    "fmt::",
    "pybind11::",
    "{_",
    "void __",
    "nvtx",
    "llvm.",
    'operator""',
    # "nlohmann::"
]


#####################################################
# remove excluded Nodes
#####################################################


_gbody = []
for path in Path(".").rglob("*.filt"):
    gfile = graphviz.Source.from_file(path.absolute()).source.split("\n")[3:-2]

    for i in range(len(gfile)):
        if "main" in gfile[i]:
            gfile[i] = gfile[i].replace("main", path.name.split(".")[0])

            print(gfile[i])

    _gbody += gfile

with open("_body_tmp", "w") as fout:
    for l in _gbody:
        fout.write(l)
        fout.write("\n")

import re

pattern = r"\s*Node\d+x[0-9a-fA-F]+ -> \s*Node\d+x[0-9a-fA-F]+;"

Nodes_links = []
Nodes_desc = []
for line in _gbody:
    if re.match(pattern, line):
        Nodes_links.append(line.strip())
    else:
        Nodes_desc.append(line.strip())

with open("link", "w") as fout:
    for l in Nodes_links:
        fout.write(l)
        fout.write("\n")

with open("Nodes", "w") as fout:
    for l in Nodes_desc:
        fout.write(l)
        fout.write("\n")


def is_excluded(l):
    for i in excludelist:
        if i in l:
            return True
    return False


removed_Nodes = []
filtered_Nodes = []
for l in Nodes_desc:
    if is_excluded(l):
        nd = l[l.find("Node") : l.find("Node") + 18]
        removed_Nodes.append(nd)
    else:
        filtered_Nodes.append(l)


with open("Nodes_filtered", "w") as fout:
    for l in filtered_Nodes:
        fout.write(l)
        fout.write("\n")


def is_con_excluded(l):
    for i in removed_Nodes:
        if i in l:
            return True

    return False


def identify_lines(arr):
    pattern = r"\s*Node(\d+x[0-9a-fA-F]+) -> \s*Node(\d+x[0-9a-fA-F]+);"
    matching_lines = []
    for line in arr:
        match = re.match(pattern, line)
        if match:
            source_Node = match.group(1)
            target_Node = match.group(2)
            matching_lines.append((source_Node, target_Node))
    return matching_lines


removed_hash = []
for n in removed_Nodes:
    removed_hash.append(n[4:])

link_hash = identify_lines(Nodes_links)
removed_hash = set(removed_hash)

filtered_links = []

from tqdm import tqdm

# check excluded connections
t = tqdm(total=len(Nodes_links))  # Initialise

for l, hashes in zip(Nodes_links, link_hash):
    a, b = hashes
    if (a in removed_hash) or (b in removed_hash):
        ...
    else:
        filtered_links.append(l)

    t.update(1)
t.close()


_gbody = []
for l in filtered_Nodes:
    _gbody.append(l)

for l in filtered_links:
    _gbody.append(l)


with open("tmp.dot", "w") as fout:
    for l in filtered_links:
        fout.write(l)
        fout.write("\n")
    for l in filtered_Nodes:
        fout.write(l)
        fout.write("\n")


# find duplicated nodes
# TODO

#####################################################
# output & rendering
#####################################################
g = graphviz.Digraph(
    "callgraph",
    filename="callgraph.dot",
    engine="dot",
    body=['rankdir="LR";newrank=true;'] + _gbody,
)

# print(g)

g.render()

exit()

#####################################################
# merge Nodes
#####################################################
affect_dic = {}
inverse_affect_dic = {}
replace_dic = {}
line_to_rm = []


def get_label(l):
    startidx = l.find(r'label="{') + 8
    endidx = l[startidx:].find(r'}"')
    return l[startidx : startidx + endidx]


for l in _gbody:
    if "label=" in l:
        lbl = get_label(l)

        if lbl == "main":
            continue

        nd = l[l.find("Node") : l.find("Node") + 18]

        if lbl in affect_dic.keys():
            replace_dic[nd] = affect_dic[lbl]
            line_to_rm.append(l)
        else:
            affect_dic[lbl] = nd
            inverse_affect_dic[nd] = lbl

print(affect_dic)

print(replace_dic)

print(inverse_affect_dic.keys())


for l in line_to_rm:
    nd = l[l.find("Node") : l.find("Node") + 18]
    print(nd, nd in inverse_affect_dic.keys())

    _gbody.remove(l)


for key in replace_dic.keys():
    new_nd = replace_dic[key]
    old_nd = key

    print(new_nd, old_nd)

    for i in range(len(_gbody)):
        _gbody[i] = _gbody[i].replace(old_nd, new_nd)


#####################################################
# restrict to used Nodes
#####################################################

Nodes = {}
links = {}

for l in _gbody:
    if "label=" in l:
        nd = l[l.find("Node") : l.find("Node") + 18]
        Nodes[nd] = l[l.find(" [") + 1 : l.find(";")]
    elif " -> " in l:
        tmp = l.split(" -> ")

        nd1 = tmp[0][tmp[0].find("Node") : tmp[0].find("Node") + 18]
        nd2 = tmp[1][tmp[1].find("Node") : tmp[1].find("Node") + 18]

        if nd1 in links.keys():
            links[nd1] += [nd2]
        else:
            links[nd1] = [nd2]
    else:
        print(l)


print(Nodes)
print(links)

# get_starting_Node


def get_starting_Node(Node_label):
    start_Node_id = ""
    for k in Nodes.keys():
        if Node_label == get_label(Nodes[k]):
            start_Node_id = k
    return start_Node_id


_gbody = []
# remake body dot graph

used_Nodes = {}
used_links = {}


def add_childs(Node_id, depth=0):
    if depth > 10:
        return

    if depth < 3:
        print(Nodes[Node_id])

    used_Nodes[Node_id] = True

    if Node_id in links.keys():
        for child_Node_id in links[Node_id]:
            used_links[Node_id + " -> " + child_Node_id] = True

            add_childs(child_Node_id, depth + 1)


# print('add_childs(get_starting_Node("main_test"))')
# add_childs(get_starting_Node("main_test"))

# print('add_childs(get_starting_Node("main_amr"))')
# add_childs(get_starting_Node("main_amr"))

print('add_childs(get_starting_Node("main"))')
add_childs(get_starting_Node("main"))

# print('add_childs(get_starting_Node("main_visu"))')
# add_childs(get_starting_Node("main_visu"))


# for n in Nodes:
#     print(Nodes[n])

#     if get_label(Nodes[n]).startswith("typeinfo name for "):
#         Nodes[n] = Nodes[n].replace("typeinfo name for ","")

#         print('add_childs('+n+')')
#         add_childs(n)


#####################################################
# rebuild graph
#####################################################


list_std = [
    "printf",
    "vfprintf",
    "vsnprintf",
    "vsprintf",
    "fclose",
    "exit",
    "system",
    "operator new(unsigned long)",
    "operator delete(void*)",
    "operator new[](unsigned long)",
    "operator delete[](void*)",
    "labs",
]


def is_std_group(Node_label):
    if Node_label in list_std:
        return True

    if Node_label.startswith("std::"):
        return True

    return False


def is_run_tests_group(Node_label):
    if Node_label.startswith("run_tests"):
        return True

    return False


def is_unittest_group(Node_label):
    if Node_label.startswith("unit_test::"):
        return True

    return False


for nid in used_Nodes.keys():
    if get_label(Nodes[nid]).startswith("MPI_"):
        print(nid)
    elif is_std_group(get_label(Nodes[nid])):
        print(nid)
    elif is_run_tests_group(get_label(Nodes[nid])):
        print(nid)
    elif is_unittest_group(get_label(Nodes[nid])):
        print(nid)
    else:
        _gbody.append(nid + " " + Nodes[nid] + ";\n")

# mpi cluster
_gbody += ['subgraph clustermpi_subgraph { style=filled; rank=max;label = "MPI instructions";  ']
for nid in used_Nodes.keys():
    if get_label(Nodes[nid]).startswith("MPI_"):
        _gbody.append(nid + ' [style=filled;color=cyan1;label="' + get_label(Nodes[nid]) + '"];\n')
_gbody += ["}"]


# std group cluster
_gbody += [
    'subgraph clusterstdgroup_subgraph { style=filled; rank=max;label = "std instructions";  '
]
for nid in used_Nodes.keys():
    if is_std_group(get_label(Nodes[nid])):
        if get_label(Nodes[nid]) == "exit":
            _gbody.append(
                nid + ' [style=filled;color=red;label="' + get_label(Nodes[nid]) + '"];\n'
            )
        else:
            _gbody.append(
                nid + ' [style=filled;color=bisque;label="' + get_label(Nodes[nid]) + '"];\n'
            )
_gbody += ["}"]


# std run_tests cluster
_gbody += ['subgraph clusterrun_testsgroup_subgraph { style=filled;label = "run_tests";  ']
for nid in used_Nodes.keys():
    if is_run_tests_group(get_label(Nodes[nid])):
        _gbody.append(nid + ' [style=filled;color=white;label="' + get_label(Nodes[nid]) + '"];\n')
_gbody += ["}"]


# std unit_tests cluster
_gbody += ['subgraph clusterunit_testsgroup_subgraph { style=filled;label = "unit_test::";  ']
for nid in used_Nodes.keys():
    if is_unittest_group(get_label(Nodes[nid])):
        _gbody.append(nid + ' [style=filled;color=white;label="' + get_label(Nodes[nid]) + '"];\n')
_gbody += ["}"]


for lk in used_links.keys():
    _gbody.append(lk + ";\n")


#####################################################
# modify look mains
#####################################################

rep_stmap = {
    "main_sph": "fillcolor=green , style=filled ",
    "main_amr": "fillcolor=green , style=filled ",
    "main_test": "fillcolor=green , style=filled ",
    "main_visu": "fillcolor=green , style=filled ",
}


move_map = {
    "main_sph": "rank=min; ",
    "main_amr": "rank=min; ",
    "main_test": "rank=min; ",
    "main_visu": "rank=min; ",
}


for kk in rep_stmap.keys():
    for i in range(len(_gbody)):
        if get_label(_gbody[i]) == kk:
            _gbody[i] = _gbody[i].replace("shape=record", rep_stmap[kk])

            print(_gbody[i])


for kk in move_map.keys():
    for i in range(len(_gbody)):
        if get_label(_gbody[i]) == kk:
            _gbody[i] = "{\n    " + move_map[kk] + _gbody[i].replace("{" + kk + "}", kk) + "}\n"

            print(_gbody[i])


#####################################################
# output & rendering
#####################################################
g = graphviz.Digraph(
    "callgraph",
    filename="callgraph.dot",
    engine="dot",
    body=['rankdir="LR";newrank=true;'] + _gbody,
)

# print(g)

g.render()
