// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file MpiWrapper.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This header does the MPI include and wrap MPI calls
 *
 */

#include "shamcomm/mpi.hpp"

// #define MPI_LOGGER_ENABLED

#ifdef MPI_LOGGER_ENABLED
    // https://stackoverflow.com/questions/6245735/pretty-print-stdtuple
    #include <iostream>
    #include <string>
    #include <tuple>
    #include <utility>
    #include <vector>

template<class TupType, size_t... I>
inline void __print_tuple(const TupType &_tup, std::index_sequence<I...>) {
    std::cout << "(";
    (..., (std::cout << (I == 0 ? "" : ", ") << std::get<I>(_tup)));
    std::cout << ")\n";
}

template<class... T>
inline void __print_tuple(const std::tuple<T...> &_tup) {
    __print_tuple(_tup, std::make_index_sequence<sizeof...(T)>());
}
    #define CALL_LOG_RETURN(a, b)                                                                  \
        std::cout << "%MPI_TRACE:" << #a;                                                          \
        __print_tuple(std::make_tuple b);                                                          \
        return a b

#else
    #define CALL_LOG_RETURN(a, b) return a b
#endif

namespace mpi {

    inline int abort(MPI_Comm comm, int errorcode) {
        CALL_LOG_RETURN(MPI_Abort, (comm, errorcode));
    }
    inline int accumulate(
        const void *origin_addr,
        int origin_count,
        MPI_Datatype origin_datatype,
        int target_rank,
        MPI_Aint target_disp,
        int target_count,
        MPI_Datatype target_datatype,
        MPI_Op op,
        MPI_Win win) {
        CALL_LOG_RETURN(
            MPI_Accumulate,
            (origin_addr,
             origin_count,
             origin_datatype,
             target_rank,
             target_disp,
             target_count,
             target_datatype,
             op,
             win));
    }
    inline int add_error_class(int *errorclass) {
        CALL_LOG_RETURN(MPI_Add_error_class, (errorclass));
    }
    inline int add_error_code(int errorclass, int *errorcode) {
        CALL_LOG_RETURN(MPI_Add_error_code, (errorclass, errorcode));
    }
    inline int add_error_string(int errorcode, const char *string) {
        CALL_LOG_RETURN(MPI_Add_error_string, (errorcode, string));
    }
    inline int alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr) {
        CALL_LOG_RETURN(MPI_Alloc_mem, (size, info, baseptr));
    }
    inline int iallreduce(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Iallreduce, (sendbuf, recvbuf, count, datatype, op, comm, request));
    }
    inline int alltoall(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Alltoall, (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
    }
    inline int ialltoall(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ialltoall,
            (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request));
    }
    inline int alltoallv(
        const void *sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Alltoallv,
            (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm));
    }
    inline int ialltoallv(
        const void *sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ialltoallv,
            (sendbuf,
             sendcounts,
             sdispls,
             sendtype,
             recvbuf,
             recvcounts,
             rdispls,
             recvtype,
             comm,
             request));
    }
    inline int alltoallw(
        const void *sendbuf,
        const int sendcounts[],
        const int sdispls[],
        const MPI_Datatype sendtypes[],
        void *recvbuf,
        const int recvcounts[],
        const int rdispls[],
        const MPI_Datatype recvtypes[],
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Alltoallw,
            (sendbuf,
             sendcounts,
             sdispls,
             sendtypes,
             recvbuf,
             recvcounts,
             rdispls,
             recvtypes,
             comm));
    }
    inline int ialltoallw(
        const void *sendbuf,
        const int sendcounts[],
        const int sdispls[],
        const MPI_Datatype sendtypes[],
        void *recvbuf,
        const int recvcounts[],
        const int rdispls[],
        const MPI_Datatype recvtypes[],
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ialltoallw,
            (sendbuf,
             sendcounts,
             sdispls,
             sendtypes,
             recvbuf,
             recvcounts,
             rdispls,
             recvtypes,
             comm,
             request));
    }
    inline int barrier(MPI_Comm comm) { CALL_LOG_RETURN(MPI_Barrier, (comm)); }
    inline int ibarrier(MPI_Comm comm, MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Ibarrier, (comm, request));
    }
    inline int bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Bcast, (buffer, count, datatype, root, comm));
    }
    inline int bsend(
        const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Bsend, (buf, count, datatype, dest, tag, comm));
    }
    inline int ibcast(
        void *buffer,
        int count,
        MPI_Datatype datatype,
        int root,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Ibcast, (buffer, count, datatype, root, comm, request));
    }
    inline int bsend_init(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Bsend_init, (buf, count, datatype, dest, tag, comm, request));
    }
    inline int buffer_attach(void *buffer, int size) {
        CALL_LOG_RETURN(MPI_Buffer_attach, (buffer, size));
    }
    inline int buffer_detach(void *buffer, int *size) {
        CALL_LOG_RETURN(MPI_Buffer_detach, (buffer, size));
    }
    inline int cancel(MPI_Request *request) { CALL_LOG_RETURN(MPI_Cancel, (request)); }
    inline int cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]) {
        CALL_LOG_RETURN(MPI_Cart_coords, (comm, rank, maxdims, coords));
    }
    inline int cart_create(
        MPI_Comm old_comm,
        int ndims,
        const int dims[],
        const int periods[],
        int reorder,
        MPI_Comm *comm_cart) {
        CALL_LOG_RETURN(MPI_Cart_create, (old_comm, ndims, dims, periods, reorder, comm_cart));
    }
    inline int cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]) {
        CALL_LOG_RETURN(MPI_Cart_get, (comm, maxdims, dims, periods, coords));
    }
    inline int cart_map(
        MPI_Comm comm, int ndims, const int dims[], const int periods[], int *newrank) {
        CALL_LOG_RETURN(MPI_Cart_map, (comm, ndims, dims, periods, newrank));
    }
    inline int cart_rank(MPI_Comm comm, const int coords[], int *rank) {
        CALL_LOG_RETURN(MPI_Cart_rank, (comm, coords, rank));
    }
    inline int cart_shift(
        MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest) {
        CALL_LOG_RETURN(MPI_Cart_shift, (comm, direction, disp, rank_source, rank_dest));
    }
    inline int cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *new_comm) {
        CALL_LOG_RETURN(MPI_Cart_sub, (comm, remain_dims, new_comm));
    }
    inline int cartdim_get(MPI_Comm comm, int *ndims) {
        CALL_LOG_RETURN(MPI_Cartdim_get, (comm, ndims));
    }
    inline int close_port(const char *port_name) { CALL_LOG_RETURN(MPI_Close_port, (port_name)); }
    inline int comm_accept(
        const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm) {
        CALL_LOG_RETURN(MPI_Comm_accept, (port_name, info, root, comm, newcomm));
    }
    inline MPI_Fint comm_c2f(MPI_Comm comm) { CALL_LOG_RETURN(MPI_Comm_c2f, (comm)); }
    inline int comm_call_errhandler(MPI_Comm comm, int errorcode) {
        CALL_LOG_RETURN(MPI_Comm_call_errhandler, (comm, errorcode));
    }
    inline int comm_compare(MPI_Comm comm1, MPI_Comm comm2, int *result) {
        CALL_LOG_RETURN(MPI_Comm_compare, (comm1, comm2, result));
    }
    inline int comm_connect(
        const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm) {
        CALL_LOG_RETURN(MPI_Comm_connect, (port_name, info, root, comm, newcomm));
    }
    inline int comm_create_errhandler(
        MPI_Comm_errhandler_function *function, MPI_Errhandler *errhandler) {
        CALL_LOG_RETURN(MPI_Comm_create_errhandler, (function, errhandler));
    }
    inline int comm_create_keyval(
        MPI_Comm_copy_attr_function *comm_copy_attr_fn,
        MPI_Comm_delete_attr_function *comm_delete_attr_fn,
        int *comm_keyval,
        void *extra_state) {
        CALL_LOG_RETURN(
            MPI_Comm_create_keyval,
            (comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state));
    }
    inline int comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm) {
        CALL_LOG_RETURN(MPI_Comm_create_group, (comm, group, tag, newcomm));
    }
    inline int comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm) {
        CALL_LOG_RETURN(MPI_Comm_create, (comm, group, newcomm));
    }
    inline int comm_delete_attr(MPI_Comm comm, int comm_keyval) {
        CALL_LOG_RETURN(MPI_Comm_delete_attr, (comm, comm_keyval));
    }
    inline int comm_disconnect(MPI_Comm *comm) { CALL_LOG_RETURN(MPI_Comm_disconnect, (comm)); }
    inline int comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
        CALL_LOG_RETURN(MPI_Comm_dup, (comm, newcomm));
    }
    inline int comm_idup(MPI_Comm comm, MPI_Comm *newcomm, MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Comm_idup, (comm, newcomm, request));
    }
    inline int comm_dup_with_info(MPI_Comm comm, MPI_Info info, MPI_Comm *newcomm) {
        CALL_LOG_RETURN(MPI_Comm_dup_with_info, (comm, info, newcomm));
    }
    inline MPI_Comm comm_f2c(MPI_Fint comm) { CALL_LOG_RETURN(MPI_Comm_f2c, (comm)); }
    inline int comm_free_keyval(int *comm_keyval) {
        CALL_LOG_RETURN(MPI_Comm_free_keyval, (comm_keyval));
    }
    inline int comm_free(MPI_Comm *comm) { CALL_LOG_RETURN(MPI_Comm_free, (comm)); }
    inline int comm_get_attr(MPI_Comm comm, int comm_keyval, void *attribute_val, int *flag) {
        CALL_LOG_RETURN(MPI_Comm_get_attr, (comm, comm_keyval, attribute_val, flag));
    }
    inline int dist_graph_create(
        MPI_Comm comm_old,
        int n,
        const int nodes[],
        const int degrees[],
        const int targets[],
        const int weights[],
        MPI_Info info,
        int reorder,
        MPI_Comm *newcomm) {
        CALL_LOG_RETURN(
            MPI_Dist_graph_create,
            (comm_old, n, nodes, degrees, targets, weights, info, reorder, newcomm));
    }
    inline int dist_graph_create_adjacent(
        MPI_Comm comm_old,
        int indegree,
        const int sources[],
        const int sourceweights[],
        int outdegree,
        const int destinations[],
        const int destweights[],
        MPI_Info info,
        int reorder,
        MPI_Comm *comm_dist_graph) {
        CALL_LOG_RETURN(
            MPI_Dist_graph_create_adjacent,
            (comm_old,
             indegree,
             sources,
             sourceweights,
             outdegree,
             destinations,
             destweights,
             info,
             reorder,
             comm_dist_graph));
    }
    inline int dist_graph_neighbors(
        MPI_Comm comm,
        int maxindegree,
        int sources[],
        int sourceweights[],
        int maxoutdegree,
        int destinations[],
        int destweights[]) {
        CALL_LOG_RETURN(
            MPI_Dist_graph_neighbors,
            (comm, maxindegree, sources, sourceweights, maxoutdegree, destinations, destweights));
    }
    inline int dist_graph_neighbors_count(
        MPI_Comm comm, int *inneighbors, int *outneighbors, int *weighted) {
        CALL_LOG_RETURN(
            MPI_Dist_graph_neighbors_count, (comm, inneighbors, outneighbors, weighted));
    }
    inline int comm_get_errhandler(MPI_Comm comm, MPI_Errhandler *erhandler) {
        CALL_LOG_RETURN(MPI_Comm_get_errhandler, (comm, erhandler));
    }
    inline int comm_get_info(MPI_Comm comm, MPI_Info *info_used) {
        CALL_LOG_RETURN(MPI_Comm_get_info, (comm, info_used));
    }
    inline int comm_get_name(MPI_Comm comm, char *comm_name, int *resultlen) {
        CALL_LOG_RETURN(MPI_Comm_get_name, (comm, comm_name, resultlen));
    }
    inline int comm_get_parent(MPI_Comm *parent) { CALL_LOG_RETURN(MPI_Comm_get_parent, (parent)); }
    inline int comm_group(MPI_Comm comm, MPI_Group *group) {
        CALL_LOG_RETURN(MPI_Comm_group, (comm, group));
    }
    inline int comm_join(int fd, MPI_Comm *intercomm) {
        CALL_LOG_RETURN(MPI_Comm_join, (fd, intercomm));
    }
    inline int comm_rank(MPI_Comm comm, int *rank) { CALL_LOG_RETURN(MPI_Comm_rank, (comm, rank)); }
    inline int comm_remote_group(MPI_Comm comm, MPI_Group *group) {
        CALL_LOG_RETURN(MPI_Comm_remote_group, (comm, group));
    }
    inline int comm_remote_size(MPI_Comm comm, int *size) {
        CALL_LOG_RETURN(MPI_Comm_remote_size, (comm, size));
    }
    inline int comm_set_attr(MPI_Comm comm, int comm_keyval, void *attribute_val) {
        CALL_LOG_RETURN(MPI_Comm_set_attr, (comm, comm_keyval, attribute_val));
    }
    inline int comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler) {
        CALL_LOG_RETURN(MPI_Comm_set_errhandler, (comm, errhandler));
    }
    inline int comm_set_info(MPI_Comm comm, MPI_Info info) {
        CALL_LOG_RETURN(MPI_Comm_set_info, (comm, info));
    }
    inline int comm_set_name(MPI_Comm comm, const char *comm_name) {
        CALL_LOG_RETURN(MPI_Comm_set_name, (comm, comm_name));
    }
    inline int comm_size(MPI_Comm comm, int *size) { CALL_LOG_RETURN(MPI_Comm_size, (comm, size)); }
    inline int comm_spawn(
        const char *command,
        char *argv[],
        int maxprocs,
        MPI_Info info,
        int root,
        MPI_Comm comm,
        MPI_Comm *intercomm,
        int array_of_errcodes[]) {
        CALL_LOG_RETURN(
            MPI_Comm_spawn,
            (command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes));
    }
    inline int comm_spawn_multiple(
        int count,
        char *array_of_commands[],
        char **array_of_argv[],
        const int array_of_maxprocs[],
        const MPI_Info array_of_info[],
        int root,
        MPI_Comm comm,
        MPI_Comm *intercomm,
        int array_of_errcodes[]) {
        CALL_LOG_RETURN(
            MPI_Comm_spawn_multiple,
            (count,
             array_of_commands,
             array_of_argv,
             array_of_maxprocs,
             array_of_info,
             root,
             comm,
             intercomm,
             array_of_errcodes));
    }
    inline int comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
        CALL_LOG_RETURN(MPI_Comm_split, (comm, color, key, newcomm));
    }
    inline int comm_split_type(
        MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm) {
        CALL_LOG_RETURN(MPI_Comm_split_type, (comm, split_type, key, info, newcomm));
    }
    inline int comm_test_inter(MPI_Comm comm, int *flag) {
        CALL_LOG_RETURN(MPI_Comm_test_inter, (comm, flag));
    }
    inline int compare_and_swap(
        const void *origin_addr,
        const void *compare_addr,
        void *result_addr,
        MPI_Datatype datatype,
        int target_rank,
        MPI_Aint target_disp,
        MPI_Win win) {
        CALL_LOG_RETURN(
            MPI_Compare_and_swap,
            (origin_addr, compare_addr, result_addr, datatype, target_rank, target_disp, win));
    }
    inline int dims_create(int nnodes, int ndims, int dims[]) {
        CALL_LOG_RETURN(MPI_Dims_create, (nnodes, ndims, dims));
    }
    inline MPI_Fint errhandler_c2f(MPI_Errhandler errhandler) {
        CALL_LOG_RETURN(MPI_Errhandler_c2f, (errhandler));
    }
    inline MPI_Errhandler errhandler_f2c(MPI_Fint errhandler) {
        CALL_LOG_RETURN(MPI_Errhandler_f2c, (errhandler));
    }
    inline int errhandler_free(MPI_Errhandler *errhandler) {
        CALL_LOG_RETURN(MPI_Errhandler_free, (errhandler));
    }
    inline int error_class(int errorcode, int *errorclass) {
        CALL_LOG_RETURN(MPI_Error_class, (errorcode, errorclass));
    }
    inline int error_string(int errorcode, char *string, int *resultlen) {
        CALL_LOG_RETURN(MPI_Error_string, (errorcode, string, resultlen));
    }
    inline int exscan(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Exscan, (sendbuf, recvbuf, count, datatype, op, comm));
    }
    inline int fetch_and_op(
        const void *origin_addr,
        void *result_addr,
        MPI_Datatype datatype,
        int target_rank,
        MPI_Aint target_disp,
        MPI_Op op,
        MPI_Win win) {
        CALL_LOG_RETURN(
            MPI_Fetch_and_op,
            (origin_addr, result_addr, datatype, target_rank, target_disp, op, win));
    }
    inline int iexscan(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Iexscan, (sendbuf, recvbuf, count, datatype, op, comm, request));
    }
    inline MPI_Fint file_c2f(MPI_File file) { CALL_LOG_RETURN(MPI_File_c2f, (file)); }
    inline MPI_File file_f2c(MPI_Fint file) { CALL_LOG_RETURN(MPI_File_f2c, (file)); }
    inline int file_call_errhandler(MPI_File fh, int errorcode) {
        CALL_LOG_RETURN(MPI_File_call_errhandler, (fh, errorcode));
    }
    inline int file_create_errhandler(
        MPI_File_errhandler_function *function, MPI_Errhandler *errhandler) {
        CALL_LOG_RETURN(MPI_File_create_errhandler, (function, errhandler));
    }
    inline int file_set_errhandler(MPI_File file, MPI_Errhandler errhandler) {
        CALL_LOG_RETURN(MPI_File_set_errhandler, (file, errhandler));
    }
    inline int file_get_errhandler(MPI_File file, MPI_Errhandler *errhandler) {
        CALL_LOG_RETURN(MPI_File_get_errhandler, (file, errhandler));
    }
    inline int file_delete(const char *filename, MPI_Info info) {
        CALL_LOG_RETURN(MPI_File_delete, (filename, info));
    }
    inline int file_set_size(MPI_File fh, MPI_Offset size) {
        CALL_LOG_RETURN(MPI_File_set_size, (fh, size));
    }
    inline int file_preallocate(MPI_File fh, MPI_Offset size) {
        CALL_LOG_RETURN(MPI_File_preallocate, (fh, size));
    }
    inline int file_get_size(MPI_File fh, MPI_Offset *size) {
        CALL_LOG_RETURN(MPI_File_get_size, (fh, size));
    }
    inline int file_get_group(MPI_File fh, MPI_Group *group) {
        CALL_LOG_RETURN(MPI_File_get_group, (fh, group));
    }
    inline int file_get_amode(MPI_File fh, int *amode) {
        CALL_LOG_RETURN(MPI_File_get_amode, (fh, amode));
    }
    inline int file_set_info(MPI_File fh, MPI_Info info) {
        CALL_LOG_RETURN(MPI_File_set_info, (fh, info));
    }
    inline int file_get_info(MPI_File fh, MPI_Info *info_used) {
        CALL_LOG_RETURN(MPI_File_get_info, (fh, info_used));
    }
    inline int file_get_view(
        MPI_File fh, MPI_Offset *disp, MPI_Datatype *etype, MPI_Datatype *filetype, char *datarep) {
        CALL_LOG_RETURN(MPI_File_get_view, (fh, disp, etype, filetype, datarep));
    }
    inline int file_read_at_all(
        MPI_File fh,
        MPI_Offset offset,
        void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_read_at_all, (fh, offset, buf, count, datatype, status));
    }
    inline int file_write_at_all(
        MPI_File fh,
        MPI_Offset offset,
        const void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_write_at_all, (fh, offset, buf, count, datatype, status));
    }
    inline int file_iread_at(
        MPI_File fh,
        MPI_Offset offset,
        void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iread_at, (fh, offset, buf, count, datatype, request));
    }
    inline int file_iwrite_at(
        MPI_File fh,
        MPI_Offset offset,
        const void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iwrite_at, (fh, offset, buf, count, datatype, request));
    }
    inline int file_iread_at_all(
        MPI_File fh,
        MPI_Offset offset,
        void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iread_at_all, (fh, offset, buf, count, datatype, request));
    }
    inline int file_iwrite_at_all(
        MPI_File fh,
        MPI_Offset offset,
        const void *buf,
        int count,
        MPI_Datatype datatype,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iwrite_at_all, (fh, offset, buf, count, datatype, request));
    }
    inline int file_read_all(
        MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_read_all, (fh, buf, count, datatype, status));
    }
    inline int file_iread(
        MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iread, (fh, buf, count, datatype, request));
    }
    inline int file_iwrite(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iwrite, (fh, buf, count, datatype, request));
    }
    inline int file_iread_all(
        MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iread_all, (fh, buf, count, datatype, request));
    }
    inline int file_iwrite_all(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iwrite_all, (fh, buf, count, datatype, request));
    }
    inline int file_seek(MPI_File fh, MPI_Offset offset, int whence) {
        CALL_LOG_RETURN(MPI_File_seek, (fh, offset, whence));
    }
    inline int file_get_position(MPI_File fh, MPI_Offset *offset) {
        CALL_LOG_RETURN(MPI_File_get_position, (fh, offset));
    }
    inline int file_get_byte_offset(MPI_File fh, MPI_Offset offset, MPI_Offset *disp) {
        CALL_LOG_RETURN(MPI_File_get_byte_offset, (fh, offset, disp));
    }
    inline int file_read_shared(
        MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_read_shared, (fh, buf, count, datatype, status));
    }
    inline int file_write_shared(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_write_shared, (fh, buf, count, datatype, status));
    }
    inline int file_iread_shared(
        MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iread_shared, (fh, buf, count, datatype, request));
    }
    inline int file_iwrite_shared(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Request *request) {
        CALL_LOG_RETURN(MPI_File_iwrite_shared, (fh, buf, count, datatype, request));
    }
    inline int file_read_ordered(
        MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_read_ordered, (fh, buf, count, datatype, status));
    }
    inline int file_write_ordered(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_write_ordered, (fh, buf, count, datatype, status));
    }
    inline int file_seek_shared(MPI_File fh, MPI_Offset offset, int whence) {
        CALL_LOG_RETURN(MPI_File_seek_shared, (fh, offset, whence));
    }
    inline int file_get_position_shared(MPI_File fh, MPI_Offset *offset) {
        CALL_LOG_RETURN(MPI_File_get_position_shared, (fh, offset));
    }
    inline int file_read_at_all_begin(
        MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype) {
        CALL_LOG_RETURN(MPI_File_read_at_all_begin, (fh, offset, buf, count, datatype));
    }
    inline int file_read_at_all_end(MPI_File fh, void *buf, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_read_at_all_end, (fh, buf, status));
    }
    inline int file_write_at_all_begin(
        MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype) {
        CALL_LOG_RETURN(MPI_File_write_at_all_begin, (fh, offset, buf, count, datatype));
    }
    inline int file_write_at_all_end(MPI_File fh, const void *buf, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_write_at_all_end, (fh, buf, status));
    }
    inline int file_read_all_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype) {
        CALL_LOG_RETURN(MPI_File_read_all_begin, (fh, buf, count, datatype));
    }
    inline int file_read_all_end(MPI_File fh, void *buf, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_read_all_end, (fh, buf, status));
    }
    inline int file_write_all_begin(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype) {
        CALL_LOG_RETURN(MPI_File_write_all_begin, (fh, buf, count, datatype));
    }
    inline int file_write_all_end(MPI_File fh, const void *buf, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_write_all_end, (fh, buf, status));
    }
    inline int file_read_ordered_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype) {
        CALL_LOG_RETURN(MPI_File_read_ordered_begin, (fh, buf, count, datatype));
    }
    inline int file_read_ordered_end(MPI_File fh, void *buf, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_read_ordered_end, (fh, buf, status));
    }
    inline int file_write_ordered_begin(
        MPI_File fh, const void *buf, int count, MPI_Datatype datatype) {
        CALL_LOG_RETURN(MPI_File_write_ordered_begin, (fh, buf, count, datatype));
    }
    inline int file_write_ordered_end(MPI_File fh, const void *buf, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_File_write_ordered_end, (fh, buf, status));
    }
    inline int file_get_type_extent(MPI_File fh, MPI_Datatype datatype, MPI_Aint *extent) {
        CALL_LOG_RETURN(MPI_File_get_type_extent, (fh, datatype, extent));
    }
    inline int file_set_atomicity(MPI_File fh, int flag) {
        CALL_LOG_RETURN(MPI_File_set_atomicity, (fh, flag));
    }
    inline int file_get_atomicity(MPI_File fh, int *flag) {
        CALL_LOG_RETURN(MPI_File_get_atomicity, (fh, flag));
    }
    inline int file_sync(MPI_File fh) { CALL_LOG_RETURN(MPI_File_sync, (fh)); }
    inline int finalize(void) { CALL_LOG_RETURN(MPI_Finalize, ()); }
    inline int finalized(int *flag) { CALL_LOG_RETURN(MPI_Finalized, (flag)); }
    inline int free_mem(void *base) { CALL_LOG_RETURN(MPI_Free_mem, (base)); }
    inline int igather(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Igather,
            (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request));
    }
    inline int igatherv(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int displs[],
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Igatherv,
            (sendbuf,
             sendcount,
             sendtype,
             recvbuf,
             recvcounts,
             displs,
             recvtype,
             root,
             comm,
             request));
    }
    inline int get_address(const void *location, MPI_Aint *address) {
        CALL_LOG_RETURN(MPI_Get_address, (location, address));
    }
    inline int get_elements(const MPI_Status *status, MPI_Datatype datatype, int *count) {
        CALL_LOG_RETURN(MPI_Get_elements, (status, datatype, count));
    }
    inline int get_elements_x(const MPI_Status *status, MPI_Datatype datatype, MPI_Count *count) {
        CALL_LOG_RETURN(MPI_Get_elements_x, (status, datatype, count));
    }
    inline int get(
        void *origin_addr,
        int origin_count,
        MPI_Datatype origin_datatype,
        int target_rank,
        MPI_Aint target_disp,
        int target_count,
        MPI_Datatype target_datatype,
        MPI_Win win) {
        CALL_LOG_RETURN(
            MPI_Get,
            (origin_addr,
             origin_count,
             origin_datatype,
             target_rank,
             target_disp,
             target_count,
             target_datatype,
             win));
    }
    inline int get_accumulate(
        const void *origin_addr,
        int origin_count,
        MPI_Datatype origin_datatype,
        void *result_addr,
        int result_count,
        MPI_Datatype result_datatype,
        int target_rank,
        MPI_Aint target_disp,
        int target_count,
        MPI_Datatype target_datatype,
        MPI_Op op,
        MPI_Win win) {
        CALL_LOG_RETURN(
            MPI_Get_accumulate,
            (origin_addr,
             origin_count,
             origin_datatype,
             result_addr,
             result_count,
             result_datatype,
             target_rank,
             target_disp,
             target_count,
             target_datatype,
             op,
             win));
    }
    inline int get_library_version(char *version, int *resultlen) {
        CALL_LOG_RETURN(MPI_Get_library_version, (version, resultlen));
    }
    inline int get_processor_name(char *name, int *resultlen) {
        CALL_LOG_RETURN(MPI_Get_processor_name, (name, resultlen));
    }
    inline int get_version(int *version, int *subversion) {
        CALL_LOG_RETURN(MPI_Get_version, (version, subversion));
    }
    inline int graph_create(
        MPI_Comm comm_old,
        int nnodes,
        const int index[],
        const int edges[],
        int reorder,
        MPI_Comm *comm_graph) {
        CALL_LOG_RETURN(MPI_Graph_create, (comm_old, nnodes, index, edges, reorder, comm_graph));
    }
    inline int graph_get(MPI_Comm comm, int maxindex, int maxedges, int index[], int edges[]) {
        CALL_LOG_RETURN(MPI_Graph_get, (comm, maxindex, maxedges, index, edges));
    }
    inline int graph_map(
        MPI_Comm comm, int nnodes, const int index[], const int edges[], int *newrank) {
        CALL_LOG_RETURN(MPI_Graph_map, (comm, nnodes, index, edges, newrank));
    }
    inline int graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors) {
        CALL_LOG_RETURN(MPI_Graph_neighbors_count, (comm, rank, nneighbors));
    }
    inline int graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int neighbors[]) {
        CALL_LOG_RETURN(MPI_Graph_neighbors, (comm, rank, maxneighbors, neighbors));
    }
    inline int graphdims_get(MPI_Comm comm, int *nnodes, int *nedges) {
        CALL_LOG_RETURN(MPI_Graphdims_get, (comm, nnodes, nedges));
    }
    inline int grequest_complete(MPI_Request request) {
        CALL_LOG_RETURN(MPI_Grequest_complete, (request));
    }
    inline int grequest_start(
        MPI_Grequest_query_function *query_fn,
        MPI_Grequest_free_function *free_fn,
        MPI_Grequest_cancel_function *cancel_fn,
        void *extra_state,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Grequest_start, (query_fn, free_fn, cancel_fn, extra_state, request));
    }
    inline MPI_Fint group_c2f(MPI_Group group) { CALL_LOG_RETURN(MPI_Group_c2f, (group)); }
    inline int group_compare(MPI_Group group1, MPI_Group group2, int *result) {
        CALL_LOG_RETURN(MPI_Group_compare, (group1, group2, result));
    }
    inline int group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup) {
        CALL_LOG_RETURN(MPI_Group_difference, (group1, group2, newgroup));
    }
    inline int group_excl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup) {
        CALL_LOG_RETURN(MPI_Group_excl, (group, n, ranks, newgroup));
    }
    inline MPI_Group group_f2c(MPI_Fint group) { CALL_LOG_RETURN(MPI_Group_f2c, (group)); }
    inline int group_free(MPI_Group *group) { CALL_LOG_RETURN(MPI_Group_free, (group)); }
    inline int group_incl(MPI_Group group, int n, const int ranks[], MPI_Group *newgroup) {
        CALL_LOG_RETURN(MPI_Group_incl, (group, n, ranks, newgroup));
    }
    inline int group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup) {
        CALL_LOG_RETURN(MPI_Group_intersection, (group1, group2, newgroup));
    }
    inline int group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup) {
        CALL_LOG_RETURN(MPI_Group_range_excl, (group, n, ranges, newgroup));
    }
    inline int group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup) {
        CALL_LOG_RETURN(MPI_Group_range_incl, (group, n, ranges, newgroup));
    }
    inline int group_rank(MPI_Group group, int *rank) {
        CALL_LOG_RETURN(MPI_Group_rank, (group, rank));
    }
    inline int group_size(MPI_Group group, int *size) {
        CALL_LOG_RETURN(MPI_Group_size, (group, size));
    }
    inline int group_translate_ranks(
        MPI_Group group1, int n, const int ranks1[], MPI_Group group2, int ranks2[]) {
        CALL_LOG_RETURN(MPI_Group_translate_ranks, (group1, n, ranks1, group2, ranks2));
    }
    inline int group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup) {
        CALL_LOG_RETURN(MPI_Group_union, (group1, group2, newgroup));
    }
    inline int ibsend(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Ibsend, (buf, count, datatype, dest, tag, comm, request));
    }
    inline int improbe(
        int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_Improbe, (source, tag, comm, flag, message, status));
    }
    inline int imrecv(
        void *buf, int count, MPI_Datatype type, MPI_Message *message, MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Imrecv, (buf, count, type, message, request));
    }
    inline MPI_Fint info_c2f(MPI_Info info) { CALL_LOG_RETURN(MPI_Info_c2f, (info)); }
    inline int info_create(MPI_Info *info) { CALL_LOG_RETURN(MPI_Info_create, (info)); }
    inline int info_delete(MPI_Info info, const char *key) {
        CALL_LOG_RETURN(MPI_Info_delete, (info, key));
    }
    inline int info_dup(MPI_Info info, MPI_Info *newinfo) {
        CALL_LOG_RETURN(MPI_Info_dup, (info, newinfo));
    }
    inline MPI_Info info_f2c(MPI_Fint info) { CALL_LOG_RETURN(MPI_Info_f2c, (info)); }
    inline int info_free(MPI_Info *info) { CALL_LOG_RETURN(MPI_Info_free, (info)); }
    inline int info_get(MPI_Info info, const char *key, int valuelen, char *value, int *flag) {
        CALL_LOG_RETURN(MPI_Info_get, (info, key, valuelen, value, flag));
    }
    inline int info_get_nkeys(MPI_Info info, int *nkeys) {
        CALL_LOG_RETURN(MPI_Info_get_nkeys, (info, nkeys));
    }
    inline int info_get_nthkey(MPI_Info info, int n, char *key) {
        CALL_LOG_RETURN(MPI_Info_get_nthkey, (info, n, key));
    }
    inline int info_get_valuelen(MPI_Info info, const char *key, int *valuelen, int *flag) {
        CALL_LOG_RETURN(MPI_Info_get_valuelen, (info, key, valuelen, flag));
    }
    inline int info_set(MPI_Info info, const char *key, const char *value) {
        CALL_LOG_RETURN(MPI_Info_set, (info, key, value));
    }
    inline int init(int *argc, char ***argv) { CALL_LOG_RETURN(MPI_Init, (argc, argv)); }
    inline int initialized(int *flag) { CALL_LOG_RETURN(MPI_Initialized, (flag)); }
    inline int init_thread(int *argc, char ***argv, int required, int *provided) {
        CALL_LOG_RETURN(MPI_Init_thread, (argc, argv, required, provided));
    }
    inline int intercomm_create(
        MPI_Comm local_comm,
        int local_leader,
        MPI_Comm bridge_comm,
        int remote_leader,
        int tag,
        MPI_Comm *newintercomm) {
        CALL_LOG_RETURN(
            MPI_Intercomm_create,
            (local_comm, local_leader, bridge_comm, remote_leader, tag, newintercomm));
    }
    inline int intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintercomm) {
        CALL_LOG_RETURN(MPI_Intercomm_merge, (intercomm, high, newintercomm));
    }
    inline int iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_Iprobe, (source, tag, comm, flag, status));
    }
    inline int irsend(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Irsend, (buf, count, datatype, dest, tag, comm, request));
    }
    inline int issend(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Issend, (buf, count, datatype, dest, tag, comm, request));
    }
    inline int is_thread_main(int *flag) { CALL_LOG_RETURN(MPI_Is_thread_main, (flag)); }
    inline int lookup_name(const char *service_name, MPI_Info info, char *port_name) {
        CALL_LOG_RETURN(MPI_Lookup_name, (service_name, info, port_name));
    }
    inline MPI_Fint message_c2f(MPI_Message message) {
        CALL_LOG_RETURN(MPI_Message_c2f, (message));
    }
    inline MPI_Message message_f2c(MPI_Fint message) {
        CALL_LOG_RETURN(MPI_Message_f2c, (message));
    }
    inline int mprobe(
        int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_Mprobe, (source, tag, comm, message, status));
    }
    inline int mrecv(
        void *buf, int count, MPI_Datatype type, MPI_Message *message, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_Mrecv, (buf, count, type, message, status));
    }
    inline int neighbor_allgather(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Neighbor_allgather,
            (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
    }
    inline int ineighbor_allgather(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ineighbor_allgather,
            (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request));
    }
    inline int neighbor_allgatherv(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int displs[],
        MPI_Datatype recvtype,
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Neighbor_allgatherv,
            (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm));
    }
    inline int ineighbor_allgatherv(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int displs[],
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ineighbor_allgatherv,
            (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request));
    }
    inline int neighbor_alltoall(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Neighbor_alltoall,
            (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm));
    }
    inline int ineighbor_alltoall(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ineighbor_alltoall,
            (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request));
    }
    inline int neighbor_alltoallv(
        const void *sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Neighbor_alltoallv,
            (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm));
    }
    inline int ineighbor_alltoallv(
        const void *sendbuf,
        const int sendcounts[],
        const int sdispls[],
        MPI_Datatype sendtype,
        void *recvbuf,
        const int recvcounts[],
        const int rdispls[],
        MPI_Datatype recvtype,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ineighbor_alltoallv,
            (sendbuf,
             sendcounts,
             sdispls,
             sendtype,
             recvbuf,
             recvcounts,
             rdispls,
             recvtype,
             comm,
             request));
    }
    inline int neighbor_alltoallw(
        const void *sendbuf,
        const int sendcounts[],
        const MPI_Aint sdispls[],
        const MPI_Datatype sendtypes[],
        void *recvbuf,
        const int recvcounts[],
        const MPI_Aint rdispls[],
        const MPI_Datatype recvtypes[],
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Neighbor_alltoallw,
            (sendbuf,
             sendcounts,
             sdispls,
             sendtypes,
             recvbuf,
             recvcounts,
             rdispls,
             recvtypes,
             comm));
    }
    inline int ineighbor_alltoallw(
        const void *sendbuf,
        const int sendcounts[],
        const MPI_Aint sdispls[],
        const MPI_Datatype sendtypes[],
        void *recvbuf,
        const int recvcounts[],
        const MPI_Aint rdispls[],
        const MPI_Datatype recvtypes[],
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ineighbor_alltoallw,
            (sendbuf,
             sendcounts,
             sdispls,
             sendtypes,
             recvbuf,
             recvcounts,
             rdispls,
             recvtypes,
             comm,
             request));
    }
    inline MPI_Fint op_c2f(MPI_Op op) { CALL_LOG_RETURN(MPI_Op_c2f, (op)); }
    inline int op_commutative(MPI_Op op, int *commute) {
        CALL_LOG_RETURN(MPI_Op_commutative, (op, commute));
    }
    inline int op_create(MPI_User_function *function, int commute, MPI_Op *op) {
        CALL_LOG_RETURN(MPI_Op_create, (function, commute, op));
    }
    inline int open_port(MPI_Info info, char *port_name) {
        CALL_LOG_RETURN(MPI_Open_port, (info, port_name));
    }
    inline MPI_Op op_f2c(MPI_Fint op) { CALL_LOG_RETURN(MPI_Op_f2c, (op)); }
    inline int op_free(MPI_Op *op) { CALL_LOG_RETURN(MPI_Op_free, (op)); }
    inline int pack_external(
        const char datarep[],
        const void *inbuf,
        int incount,
        MPI_Datatype datatype,
        void *outbuf,
        MPI_Aint outsize,
        MPI_Aint *position) {
        CALL_LOG_RETURN(
            MPI_Pack_external, (datarep, inbuf, incount, datatype, outbuf, outsize, position));
    }
    inline int pack_external_size(
        const char datarep[], int incount, MPI_Datatype datatype, MPI_Aint *size) {
        CALL_LOG_RETURN(MPI_Pack_external_size, (datarep, incount, datatype, size));
    }
    inline int pack(
        const void *inbuf,
        int incount,
        MPI_Datatype datatype,
        void *outbuf,
        int outsize,
        int *position,
        MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Pack, (inbuf, incount, datatype, outbuf, outsize, position, comm));
    }
    inline int pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size) {
        CALL_LOG_RETURN(MPI_Pack_size, (incount, datatype, comm, size));
    }
    /*
    inline int pcontrol(const int level, ...){
        CALL_LOG_RETURN(MPI_Pcontrol,(level,...));
    }*/
    inline int publish_name(const char *service_name, MPI_Info info, const char *port_name) {
        CALL_LOG_RETURN(MPI_Publish_name, (service_name, info, port_name));
    }
    inline int put(
        const void *origin_addr,
        int origin_count,
        MPI_Datatype origin_datatype,
        int target_rank,
        MPI_Aint target_disp,
        int target_count,
        MPI_Datatype target_datatype,
        MPI_Win win) {
        CALL_LOG_RETURN(
            MPI_Put,
            (origin_addr,
             origin_count,
             origin_datatype,
             target_rank,
             target_disp,
             target_count,
             target_datatype,
             win));
    }
    inline int query_thread(int *provided) { CALL_LOG_RETURN(MPI_Query_thread, (provided)); }
    inline int raccumulate(
        const void *origin_addr,
        int origin_count,
        MPI_Datatype origin_datatype,
        int target_rank,
        MPI_Aint target_disp,
        int target_count,
        MPI_Datatype target_datatype,
        MPI_Op op,
        MPI_Win win,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Raccumulate,
            (origin_addr,
             origin_count,
             origin_datatype,
             target_rank,
             target_disp,
             target_count,
             target_datatype,
             op,
             win,
             request));
    }
    inline int recv_init(
        void *buf,
        int count,
        MPI_Datatype datatype,
        int source,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Recv_init, (buf, count, datatype, source, tag, comm, request));
    }
    inline int reduce(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        int root,
        MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Reduce, (sendbuf, recvbuf, count, datatype, op, root, comm));
    }
    inline int ireduce(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        int root,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Ireduce, (sendbuf, recvbuf, count, datatype, op, root, comm, request));
    }
    inline int reduce_local(
        const void *inbuf, void *inoutbuf, int count, MPI_Datatype datatype, MPI_Op op) {
        CALL_LOG_RETURN(MPI_Reduce_local, (inbuf, inoutbuf, count, datatype, op));
    }
    inline int reduce_scatter(
        const void *sendbuf,
        void *recvbuf,
        const int recvcounts[],
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Reduce_scatter, (sendbuf, recvbuf, recvcounts, datatype, op, comm));
    }
    inline int ireduce_scatter(
        const void *sendbuf,
        void *recvbuf,
        const int recvcounts[],
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ireduce_scatter, (sendbuf, recvbuf, recvcounts, datatype, op, comm, request));
    }
    inline int reduce_scatter_block(
        const void *sendbuf,
        void *recvbuf,
        int recvcount,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Reduce_scatter_block, (sendbuf, recvbuf, recvcount, datatype, op, comm));
    }
    inline int ireduce_scatter_block(
        const void *sendbuf,
        void *recvbuf,
        int recvcount,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Ireduce_scatter_block, (sendbuf, recvbuf, recvcount, datatype, op, comm, request));
    }
    inline int register_datarep(
        const char *datarep,
        MPI_Datarep_conversion_function *read_conversion_fn,
        MPI_Datarep_conversion_function *write_conversion_fn,
        MPI_Datarep_extent_function *dtype_file_extent_fn,
        void *extra_state) {
        CALL_LOG_RETURN(
            MPI_Register_datarep,
            (datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state));
    }
    inline MPI_Fint request_c2f(MPI_Request request) {
        CALL_LOG_RETURN(MPI_Request_c2f, (request));
    }
    inline MPI_Request request_f2c(MPI_Fint request) {
        CALL_LOG_RETURN(MPI_Request_f2c, (request));
    }
    inline int request_free(MPI_Request *request) { CALL_LOG_RETURN(MPI_Request_free, (request)); }
    inline int request_get_status(MPI_Request request, int *flag, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_Request_get_status, (request, flag, status));
    }
    inline int rget(
        void *origin_addr,
        int origin_count,
        MPI_Datatype origin_datatype,
        int target_rank,
        MPI_Aint target_disp,
        int target_count,
        MPI_Datatype target_datatype,
        MPI_Win win,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Rget,
            (origin_addr,
             origin_count,
             origin_datatype,
             target_rank,
             target_disp,
             target_count,
             target_datatype,
             win,
             request));
    }
    inline int rget_accumulate(
        const void *origin_addr,
        int origin_count,
        MPI_Datatype origin_datatype,
        void *result_addr,
        int result_count,
        MPI_Datatype result_datatype,
        int target_rank,
        MPI_Aint target_disp,
        int target_count,
        MPI_Datatype target_datatype,
        MPI_Op op,
        MPI_Win win,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Rget_accumulate,
            (origin_addr,
             origin_count,
             origin_datatype,
             result_addr,
             result_count,
             result_datatype,
             target_rank,
             target_disp,
             target_count,
             target_datatype,
             op,
             win,
             request));
    }
    inline int rput(
        const void *origin_addr,
        int origin_count,
        MPI_Datatype origin_datatype,
        int target_rank,
        MPI_Aint target_disp,
        int target_cout,
        MPI_Datatype target_datatype,
        MPI_Win win,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Rput,
            (origin_addr,
             origin_count,
             origin_datatype,
             target_rank,
             target_disp,
             target_cout,
             target_datatype,
             win,
             request));
    }
    inline int rsend(
        const void *ibuf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Rsend, (ibuf, count, datatype, dest, tag, comm));
    }
    inline int rsend_init(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Rsend_init, (buf, count, datatype, dest, tag, comm, request));
    }
    inline int scan(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Scan, (sendbuf, recvbuf, count, datatype, op, comm));
    }
    inline int iscan(
        const void *sendbuf,
        void *recvbuf,
        int count,
        MPI_Datatype datatype,
        MPI_Op op,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Iscan, (sendbuf, recvbuf, count, datatype, op, comm, request));
    }
    inline int scatter(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Scatter, (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm));
    }
    inline int iscatter(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Iscatter,
            (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request));
    }
    inline int scatterv(
        const void *sendbuf,
        const int sendcounts[],
        const int displs[],
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm) {
        CALL_LOG_RETURN(
            MPI_Scatterv,
            (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm));
    }
    inline int iscatterv(
        const void *sendbuf,
        const int sendcounts[],
        const int displs[],
        MPI_Datatype sendtype,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int root,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(
            MPI_Iscatterv,
            (sendbuf,
             sendcounts,
             displs,
             sendtype,
             recvbuf,
             recvcount,
             recvtype,
             root,
             comm,
             request));
    }
    inline int send_init(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Send_init, (buf, count, datatype, dest, tag, comm, request));
    }
    inline int sendrecv(
        const void *sendbuf,
        int sendcount,
        MPI_Datatype sendtype,
        int dest,
        int sendtag,
        void *recvbuf,
        int recvcount,
        MPI_Datatype recvtype,
        int source,
        int recvtag,
        MPI_Comm comm,
        MPI_Status *status) {
        CALL_LOG_RETURN(
            MPI_Sendrecv,
            (sendbuf,
             sendcount,
             sendtype,
             dest,
             sendtag,
             recvbuf,
             recvcount,
             recvtype,
             source,
             recvtag,
             comm,
             status));
    }
    inline int sendrecv_replace(
        void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int sendtag,
        int source,
        int recvtag,
        MPI_Comm comm,
        MPI_Status *status) {
        CALL_LOG_RETURN(
            MPI_Sendrecv_replace,
            (buf, count, datatype, dest, sendtag, source, recvtag, comm, status));
    }
    inline int ssend_init(
        const void *buf,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm,
        MPI_Request *request) {
        CALL_LOG_RETURN(MPI_Ssend_init, (buf, count, datatype, dest, tag, comm, request));
    }
    inline int ssend(
        const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Ssend, (buf, count, datatype, dest, tag, comm));
    }
    inline int start(MPI_Request *request) { CALL_LOG_RETURN(MPI_Start, (request)); }
    inline int startall(int count, MPI_Request array_of_requests[]) {
        CALL_LOG_RETURN(MPI_Startall, (count, array_of_requests));
    }
    inline int status_c2f(const MPI_Status *c_status, MPI_Fint *f_status) {
        CALL_LOG_RETURN(MPI_Status_c2f, (c_status, f_status));
    }
    inline int status_f2c(const MPI_Fint *f_status, MPI_Status *c_status) {
        CALL_LOG_RETURN(MPI_Status_f2c, (f_status, c_status));
    }
    inline int status_set_cancelled(MPI_Status *status, int flag) {
        CALL_LOG_RETURN(MPI_Status_set_cancelled, (status, flag));
    }
    inline int status_set_elements(MPI_Status *status, MPI_Datatype datatype, int count) {
        CALL_LOG_RETURN(MPI_Status_set_elements, (status, datatype, count));
    }
    inline int status_set_elements_x(MPI_Status *status, MPI_Datatype datatype, MPI_Count count) {
        CALL_LOG_RETURN(MPI_Status_set_elements_x, (status, datatype, count));
    }
    inline int testall(
        int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[]) {
        CALL_LOG_RETURN(MPI_Testall, (count, array_of_requests, flag, array_of_statuses));
    }
    inline int testany(
        int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_Testany, (count, array_of_requests, index, flag, status));
    }
    inline int test(MPI_Request *request, int *flag, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_Test, (request, flag, status));
    }
    inline int test_cancelled(const MPI_Status *status, int *flag) {
        CALL_LOG_RETURN(MPI_Test_cancelled, (status, flag));
    }
    inline int testsome(
        int incount,
        MPI_Request array_of_requests[],
        int *outcount,
        int array_of_indices[],
        MPI_Status array_of_statuses[]) {
        CALL_LOG_RETURN(
            MPI_Testsome,
            (incount, array_of_requests, outcount, array_of_indices, array_of_statuses));
    }
    inline int topo_test(MPI_Comm comm, int *status) {
        CALL_LOG_RETURN(MPI_Topo_test, (comm, status));
    }
    inline MPI_Fint type_c2f(MPI_Datatype datatype) { CALL_LOG_RETURN(MPI_Type_c2f, (datatype)); }
    inline int type_commit(MPI_Datatype *type) { CALL_LOG_RETURN(MPI_Type_commit, (type)); }
    inline int type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype) {
        CALL_LOG_RETURN(MPI_Type_contiguous, (count, oldtype, newtype));
    }
    inline int type_create_darray(
        int size,
        int rank,
        int ndims,
        const int gsize_array[],
        const int distrib_array[],
        const int darg_array[],
        const int psize_array[],
        int order,
        MPI_Datatype oldtype,
        MPI_Datatype *newtype) {
        CALL_LOG_RETURN(
            MPI_Type_create_darray,
            (size,
             rank,
             ndims,
             gsize_array,
             distrib_array,
             darg_array,
             psize_array,
             order,
             oldtype,
             newtype));
    }
    inline int type_create_f90_complex(int p, int r, MPI_Datatype *newtype) {
        CALL_LOG_RETURN(MPI_Type_create_f90_complex, (p, r, newtype));
    }
    inline int type_create_f90_integer(int r, MPI_Datatype *newtype) {
        CALL_LOG_RETURN(MPI_Type_create_f90_integer, (r, newtype));
    }
    inline int type_create_f90_real(int p, int r, MPI_Datatype *newtype) {
        CALL_LOG_RETURN(MPI_Type_create_f90_real, (p, r, newtype));
    }
    inline int type_create_hindexed_block(
        int count,
        int blocklength,
        const MPI_Aint array_of_displacements[],
        MPI_Datatype oldtype,
        MPI_Datatype *newtype) {
        CALL_LOG_RETURN(
            MPI_Type_create_hindexed_block,
            (count, blocklength, array_of_displacements, oldtype, newtype));
    }
    inline int type_create_hindexed(
        int count,
        const int array_of_blocklengths[],
        const MPI_Aint array_of_displacements[],
        MPI_Datatype oldtype,
        MPI_Datatype *newtype) {
        CALL_LOG_RETURN(
            MPI_Type_create_hindexed,
            (count, array_of_blocklengths, array_of_displacements, oldtype, newtype));
    }
    inline int type_create_hvector(
        int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype) {
        CALL_LOG_RETURN(MPI_Type_create_hvector, (count, blocklength, stride, oldtype, newtype));
    }
    inline int type_create_keyval(
        MPI_Type_copy_attr_function *type_copy_attr_fn,
        MPI_Type_delete_attr_function *type_delete_attr_fn,
        int *type_keyval,
        void *extra_state) {
        CALL_LOG_RETURN(
            MPI_Type_create_keyval,
            (type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state));
    }
    inline int type_create_indexed_block(
        int count,
        int blocklength,
        const int array_of_displacements[],
        MPI_Datatype oldtype,
        MPI_Datatype *newtype) {
        CALL_LOG_RETURN(
            MPI_Type_create_indexed_block,
            (count, blocklength, array_of_displacements, oldtype, newtype));
    }
    inline int type_create_struct(
        int count,
        const int array_of_block_lengths[],
        const MPI_Aint array_of_displacements[],
        const MPI_Datatype array_of_types[],
        MPI_Datatype *newtype) {
        CALL_LOG_RETURN(
            MPI_Type_create_struct,
            (count, array_of_block_lengths, array_of_displacements, array_of_types, newtype));
    }
    inline int type_create_subarray(
        int ndims,
        const int size_array[],
        const int subsize_array[],
        const int start_array[],
        int order,
        MPI_Datatype oldtype,
        MPI_Datatype *newtype) {
        CALL_LOG_RETURN(
            MPI_Type_create_subarray,
            (ndims, size_array, subsize_array, start_array, order, oldtype, newtype));
    }
    inline int type_create_resized(
        MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype *newtype) {
        CALL_LOG_RETURN(MPI_Type_create_resized, (oldtype, lb, extent, newtype));
    }
    inline int type_delete_attr(MPI_Datatype type, int type_keyval) {
        CALL_LOG_RETURN(MPI_Type_delete_attr, (type, type_keyval));
    }
    inline int type_dup(MPI_Datatype type, MPI_Datatype *newtype) {
        CALL_LOG_RETURN(MPI_Type_dup, (type, newtype));
    }
    inline int type_free(MPI_Datatype *type) { CALL_LOG_RETURN(MPI_Type_free, (type)); }
    inline int type_free_keyval(int *type_keyval) {
        CALL_LOG_RETURN(MPI_Type_free_keyval, (type_keyval));
    }
    inline MPI_Datatype type_f2c(MPI_Fint datatype) { CALL_LOG_RETURN(MPI_Type_f2c, (datatype)); }
    inline int type_get_attr(MPI_Datatype type, int type_keyval, void *attribute_val, int *flag) {
        CALL_LOG_RETURN(MPI_Type_get_attr, (type, type_keyval, attribute_val, flag));
    }
    inline int type_get_contents(
        MPI_Datatype mtype,
        int max_integers,
        int max_addresses,
        int max_datatypes,
        int array_of_integers[],
        MPI_Aint array_of_addresses[],
        MPI_Datatype array_of_datatypes[]) {
        CALL_LOG_RETURN(
            MPI_Type_get_contents,
            (mtype,
             max_integers,
             max_addresses,
             max_datatypes,
             array_of_integers,
             array_of_addresses,
             array_of_datatypes));
    }
    inline int type_get_envelope(
        MPI_Datatype type,
        int *num_integers,
        int *num_addresses,
        int *num_datatypes,
        int *combiner) {
        CALL_LOG_RETURN(
            MPI_Type_get_envelope, (type, num_integers, num_addresses, num_datatypes, combiner));
    }
    inline int type_get_extent_x(MPI_Datatype type, MPI_Count *lb, MPI_Count *extent) {
        CALL_LOG_RETURN(MPI_Type_get_extent_x, (type, lb, extent));
    }
    inline int type_get_name(MPI_Datatype type, char *type_name, int *resultlen) {
        CALL_LOG_RETURN(MPI_Type_get_name, (type, type_name, resultlen));
    }
    inline int type_get_true_extent(
        MPI_Datatype datatype, MPI_Aint *true_lb, MPI_Aint *true_extent) {
        CALL_LOG_RETURN(MPI_Type_get_true_extent, (datatype, true_lb, true_extent));
    }
    inline int type_get_true_extent_x(
        MPI_Datatype datatype, MPI_Count *true_lb, MPI_Count *true_extent) {
        CALL_LOG_RETURN(MPI_Type_get_true_extent_x, (datatype, true_lb, true_extent));
    }
    inline int type_indexed(
        int count,
        const int array_of_blocklengths[],
        const int array_of_displacements[],
        MPI_Datatype oldtype,
        MPI_Datatype *newtype) {
        CALL_LOG_RETURN(
            MPI_Type_indexed,
            (count, array_of_blocklengths, array_of_displacements, oldtype, newtype));
    }
    inline int type_match_size(int typeclass, int size, MPI_Datatype *type) {
        CALL_LOG_RETURN(MPI_Type_match_size, (typeclass, size, type));
    }
    inline int type_set_attr(MPI_Datatype type, int type_keyval, void *attr_val) {
        CALL_LOG_RETURN(MPI_Type_set_attr, (type, type_keyval, attr_val));
    }
    inline int type_set_name(MPI_Datatype type, const char *type_name) {
        CALL_LOG_RETURN(MPI_Type_set_name, (type, type_name));
    }
    inline int type_size(MPI_Datatype type, int *size) {
        CALL_LOG_RETURN(MPI_Type_size, (type, size));
    }
    inline int type_size_x(MPI_Datatype type, MPI_Count *size) {
        CALL_LOG_RETURN(MPI_Type_size_x, (type, size));
    }
    inline int type_vector(
        int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype) {
        CALL_LOG_RETURN(MPI_Type_vector, (count, blocklength, stride, oldtype, newtype));
    }
    inline int unpack(
        const void *inbuf,
        int insize,
        int *position,
        void *outbuf,
        int outcount,
        MPI_Datatype datatype,
        MPI_Comm comm) {
        CALL_LOG_RETURN(MPI_Unpack, (inbuf, insize, position, outbuf, outcount, datatype, comm));
    }
    inline int unpublish_name(const char *service_name, MPI_Info info, const char *port_name) {
        CALL_LOG_RETURN(MPI_Unpublish_name, (service_name, info, port_name));
    }
    inline int unpack_external(
        const char datarep[],
        const void *inbuf,
        MPI_Aint insize,
        MPI_Aint *position,
        void *outbuf,
        int outcount,
        MPI_Datatype datatype) {
        CALL_LOG_RETURN(
            MPI_Unpack_external, (datarep, inbuf, insize, position, outbuf, outcount, datatype));
    }
    inline int waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status) {
        CALL_LOG_RETURN(MPI_Waitany, (count, array_of_requests, index, status));
    }

    inline int waitsome(
        int incount,
        MPI_Request array_of_requests[],
        int *outcount,
        int array_of_indices[],
        MPI_Status array_of_statuses[]) {
        CALL_LOG_RETURN(
            MPI_Waitsome,
            (incount, array_of_requests, outcount, array_of_indices, array_of_statuses));
    }
    inline int win_allocate(
        MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win) {
        CALL_LOG_RETURN(MPI_Win_allocate, (size, disp_unit, info, comm, baseptr, win));
    }
    inline int win_allocate_shared(
        MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win) {
        CALL_LOG_RETURN(MPI_Win_allocate_shared, (size, disp_unit, info, comm, baseptr, win));
    }
    inline int win_attach(MPI_Win win, void *base, MPI_Aint size) {
        CALL_LOG_RETURN(MPI_Win_attach, (win, base, size));
    }
    inline MPI_Fint win_c2f(MPI_Win win) { CALL_LOG_RETURN(MPI_Win_c2f, (win)); }
    inline int win_call_errhandler(MPI_Win win, int errorcode) {
        CALL_LOG_RETURN(MPI_Win_call_errhandler, (win, errorcode));
    }
    inline int win_complete(MPI_Win win) { CALL_LOG_RETURN(MPI_Win_complete, (win)); }
    inline int win_create(
        void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win) {
        CALL_LOG_RETURN(MPI_Win_create, (base, size, disp_unit, info, comm, win));
    }
    inline int win_create_dynamic(MPI_Info info, MPI_Comm comm, MPI_Win *win) {
        CALL_LOG_RETURN(MPI_Win_create_dynamic, (info, comm, win));
    }
    inline int win_create_errhandler(
        MPI_Win_errhandler_function *function, MPI_Errhandler *errhandler) {
        CALL_LOG_RETURN(MPI_Win_create_errhandler, (function, errhandler));
    }
    inline int win_create_keyval(
        MPI_Win_copy_attr_function *win_copy_attr_fn,
        MPI_Win_delete_attr_function *win_delete_attr_fn,
        int *win_keyval,
        void *extra_state) {
        CALL_LOG_RETURN(
            MPI_Win_create_keyval, (win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state));
    }
    inline int win_delete_attr(MPI_Win win, int win_keyval) {
        CALL_LOG_RETURN(MPI_Win_delete_attr, (win, win_keyval));
    }
    inline int win_detach(MPI_Win win, const void *base) {
        CALL_LOG_RETURN(MPI_Win_detach, (win, base));
    }
    inline MPI_Win win_f2c(MPI_Fint win) { CALL_LOG_RETURN(MPI_Win_f2c, (win)); }
    inline int win_fence(int assert, MPI_Win win) { CALL_LOG_RETURN(MPI_Win_fence, (assert, win)); }
    inline int win_flush(int rank, MPI_Win win) { CALL_LOG_RETURN(MPI_Win_flush, (rank, win)); }
    inline int win_flush_all(MPI_Win win) { CALL_LOG_RETURN(MPI_Win_flush_all, (win)); }
    inline int win_flush_local(int rank, MPI_Win win) {
        CALL_LOG_RETURN(MPI_Win_flush_local, (rank, win));
    }
    inline int win_flush_local_all(MPI_Win win) { CALL_LOG_RETURN(MPI_Win_flush_local_all, (win)); }
    inline int win_free(MPI_Win *win) { CALL_LOG_RETURN(MPI_Win_free, (win)); }
    inline int win_free_keyval(int *win_keyval) {
        CALL_LOG_RETURN(MPI_Win_free_keyval, (win_keyval));
    }
    inline int win_get_attr(MPI_Win win, int win_keyval, void *attribute_val, int *flag) {
        CALL_LOG_RETURN(MPI_Win_get_attr, (win, win_keyval, attribute_val, flag));
    }
    inline int win_get_errhandler(MPI_Win win, MPI_Errhandler *errhandler) {
        CALL_LOG_RETURN(MPI_Win_get_errhandler, (win, errhandler));
    }
    inline int win_get_group(MPI_Win win, MPI_Group *group) {
        CALL_LOG_RETURN(MPI_Win_get_group, (win, group));
    }
    inline int win_get_info(MPI_Win win, MPI_Info *info_used) {
        CALL_LOG_RETURN(MPI_Win_get_info, (win, info_used));
    }
    inline int win_get_name(MPI_Win win, char *win_name, int *resultlen) {
        CALL_LOG_RETURN(MPI_Win_get_name, (win, win_name, resultlen));
    }
    inline int win_lock(int lock_type, int rank, int assert, MPI_Win win) {
        CALL_LOG_RETURN(MPI_Win_lock, (lock_type, rank, assert, win));
    }
    inline int win_lock_all(int assert, MPI_Win win) {
        CALL_LOG_RETURN(MPI_Win_lock_all, (assert, win));
    }
    inline int win_post(MPI_Group group, int assert, MPI_Win win) {
        CALL_LOG_RETURN(MPI_Win_post, (group, assert, win));
    }
    inline int win_set_attr(MPI_Win win, int win_keyval, void *attribute_val) {
        CALL_LOG_RETURN(MPI_Win_set_attr, (win, win_keyval, attribute_val));
    }
    inline int win_set_errhandler(MPI_Win win, MPI_Errhandler errhandler) {
        CALL_LOG_RETURN(MPI_Win_set_errhandler, (win, errhandler));
    }
    inline int win_set_info(MPI_Win win, MPI_Info info) {
        CALL_LOG_RETURN(MPI_Win_set_info, (win, info));
    }
    inline int win_set_name(MPI_Win win, const char *win_name) {
        CALL_LOG_RETURN(MPI_Win_set_name, (win, win_name));
    }
    inline int win_shared_query(
        MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr) {
        CALL_LOG_RETURN(MPI_Win_shared_query, (win, rank, size, disp_unit, baseptr));
    }
    inline int win_start(MPI_Group group, int assert, MPI_Win win) {
        CALL_LOG_RETURN(MPI_Win_start, (group, assert, win));
    }
    inline int win_sync(MPI_Win win) { CALL_LOG_RETURN(MPI_Win_sync, (win)); }
    inline int win_test(MPI_Win win, int *flag) { CALL_LOG_RETURN(MPI_Win_test, (win, flag)); }
    inline int win_unlock(int rank, MPI_Win win) { CALL_LOG_RETURN(MPI_Win_unlock, (rank, win)); }
    inline int win_unlock_all(MPI_Win win) { CALL_LOG_RETURN(MPI_Win_unlock_all, (win)); }
    inline int win_wait(MPI_Win win) { CALL_LOG_RETURN(MPI_Win_wait, (win)); }
    inline double wtick(void) { CALL_LOG_RETURN(MPI_Wtick, ()); }
    inline double wtime(void) { CALL_LOG_RETURN(MPI_Wtime, ()); }

} // namespace mpi

#undef CALL_LOG_RETURN
