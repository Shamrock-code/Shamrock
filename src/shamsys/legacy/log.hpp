// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once 

/**
 * @file log.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */

#include "shambackends/typeAliasVec.hpp"
#include <chrono>
#include <fstream>
#include <string>
#include <iostream>

#include "shamsys/Log.hpp"




namespace terminal_effects {
    
    namespace impl {
        const std::string empty = "";

        const std::string esc_char = "\x1b[";

        const std::string reset = esc_char + "0m";

        const std::string bold = esc_char + "1m";
        const std::string faint = esc_char + "2m";
        const std::string underline = esc_char + "4m";
        const std::string blink = esc_char + "5m";

        namespace colors_foreground_8b {
            const std::string black =   esc_char + "30m";
            const std::string red =     esc_char + "31m";
            const std::string green =   esc_char + "32m";
            const std::string yellow =  esc_char + "33m";
            const std::string blue =    esc_char + "34m";
            const std::string magenta = esc_char + "35m";
            const std::string cyan =    esc_char + "36m";
            const std::string white =   esc_char + "37m";
        }
    }

    inline std::string empty = "";

    inline std::string esc_char = "\x1b[";

    inline std::string reset = esc_char + "0m";

    inline std::string bold = esc_char + "1m";
    inline std::string faint = esc_char + "2m";
    inline std::string underline = esc_char + "4m";
    inline std::string blink = esc_char + "5m";

    namespace colors_foreground_8b {
        inline std::string black =   esc_char + "30m";
        inline std::string red =     esc_char + "31m";
        inline std::string green =   esc_char + "32m";
        inline std::string yellow =  esc_char + "33m";
        inline std::string blue =    esc_char + "34m";
        inline std::string magenta = esc_char + "35m";
        inline std::string cyan =    esc_char + "36m";
        inline std::string white =   esc_char + "37m";
    }
    
    inline bool colored_mode = true;


    inline void enable_colors(){
        empty = impl::empty;
        esc_char = impl::esc_char;
        reset = impl::reset;
        bold = impl::bold;
        faint = impl::faint;
        underline = impl::underline;
        blink = impl::blink;
        colors_foreground_8b::black = impl::colors_foreground_8b::black;
        colors_foreground_8b::red = impl::colors_foreground_8b::red;
        colors_foreground_8b::green = impl::colors_foreground_8b::green;
        colors_foreground_8b::yellow = impl::colors_foreground_8b::yellow;
        colors_foreground_8b::blue = impl::colors_foreground_8b::blue;
        colors_foreground_8b::magenta = impl::colors_foreground_8b::magenta;
        colors_foreground_8b::cyan = impl::colors_foreground_8b::cyan;
        colors_foreground_8b::white = impl::colors_foreground_8b::white;
    }

    inline void disable_colors(){
        empty = impl::empty;
        esc_char = impl::empty;
        reset = impl::empty;
        bold = impl::empty;
        faint = impl::empty;
        underline = impl::empty;
        blink = impl::empty;
        colors_foreground_8b::black = impl::empty;
        colors_foreground_8b::red = impl::empty;
        colors_foreground_8b::green = impl::empty;
        colors_foreground_8b::yellow = impl::empty;
        colors_foreground_8b::blue = impl::empty;
        colors_foreground_8b::magenta = impl::empty;
        colors_foreground_8b::cyan = impl::empty;
        colors_foreground_8b::white = impl::empty;
    }

}




namespace logger {

    

    
    








    inline void print(){}

    template <typename T, typename... Types>
    void print(T var1, Types... var2);

    template <typename... Types>
    inline void print(std::string s, Types... var2)
    {
        std::cout << s << " ";
        print(var2...);
    }


    template <typename T,typename... Types>
    inline void print(sycl::vec<T, 2> s, Types... var2)
    {
        std::cout << shambase::format("{} ", s);
        print(var2...);
    }

    template <typename T,typename... Types>
    inline void print(sycl::vec<T, 3> s, Types... var2)
    {
        std::cout << shambase::format("{} ", s);
        print(var2...);
    }

    template <typename T,typename... Types>
    inline void print(sycl::vec<T, 4> s, Types... var2)
    {
        std::cout << shambase::format("{} ", s);
        print(var2...);
    }


    template <typename T,typename... Types>
    inline void print(sycl::vec<T, 8> s, Types... var2)
    {
        std::cout << shambase::format("{} ", s);
        print(var2...);
    }

    template <typename T,typename... Types>
    inline void print(sycl::vec<T, 16> s, Types... var2)
    {
        std::cout << shambase::format("{} ", s);
        print(var2...);
    }

    template <typename Ta, typename Tb,typename... Types>
    inline void print(std::pair<Ta,Tb> s, Types... var2)
    {
        std::cout << "(" ;
        print(s.first);
        std::cout << ",";
        print(s.second);
        std::cout << ") ";
        print(var2...);
    }


    template <class... Typestuple,typename... Types>
    inline void print(std::tuple<Typestuple...> t , Types... var2)
    {
        std::apply
        (
            [](Typestuple ... tupleArgs)
            {
                std::cout << "[ ";
                (print(std::forward<Typestuple>(tupleArgs)),...);
                std::cout << "]";
            }, t
        );
        print(var2...);
    }




    template <typename T, typename... Types>
    inline void print(T var1, Types... var2)
    {
        std::cout << var1 << " ";
        print(var2...);
    }

    
 

    




    template <typename... Types>
    inline void raw(Types... var2){
        print(var2...);
    }

    template <typename... Types>
    inline void raw_ln(Types... var2){
        raw(var2...);
        std::cout << std::endl;
    }


    inline void print_faint_row(){
        raw_ln(terminal_effects::faint + "-----------------------------------------------------" + terminal_effects::reset);
    }

    inline i8 loglevel = 0;

    #define LIST_LEVEL                                                                                                      \
    X(debug_alloc, terminal_effects::colors_foreground_8b::red, "Debug Alloc ", 127)                                     \
    X(debug_mpi, terminal_effects::colors_foreground_8b::blue, "Debug MPI ", 100)                                      \
    X(debug_sycl, terminal_effects::colors_foreground_8b::magenta, "Debug SYCL", 11)                                        \
    X(debug, terminal_effects::colors_foreground_8b::green, "Debug ", 10)                                                   \
    X(info, terminal_effects::colors_foreground_8b::cyan, "", 1)                                                            \
    X(normal, terminal_effects::bold, "", 0)                                                                                \
    X(warn, terminal_effects::colors_foreground_8b::yellow, "Warning ", -1)                                                 \
    X(err, terminal_effects::colors_foreground_8b::red, "Error ", -10)



    //////////////////////////////
    //declare all the log levels
    //////////////////////////////
    #define DECLARE_LOG_LEVEL(_name, color, loginf, logval)                                                                     \
                                                                                                                            \
    constexpr i8 log_##_name = (logval);                                                                                              \
                                                                                                                            \
    template <typename... Types> inline void _name(std::string module_name, Types... var2) {                                \
        if (loglevel >= log_##_name) {                                                                                      \
            std::cout << "[" + (color) + module_name + terminal_effects::reset + "] " + (color) + (loginf) +          \
                             terminal_effects::reset + ": ";                                                                \
            print(var2...);                                                                                                 \
        }                                                                                                                   \
    }                                                                                                                       \
                                                                                                                            \
    template <typename... Types> inline void _name##_ln(std::string module_name, Types... var2) {                           \
        if (loglevel >= log_##_name) {                                                                                      \
            _name(module_name, var2...);                                                                                    \
            std::cout << std::endl;                                                                                         \
        }                                                                                                                   \
    }

    #define X DECLARE_LOG_LEVEL
    LIST_LEVEL
    #undef X

    #undef DECLARE_LOG_LEVEL
    ///////////////////////////////////
    // log level declared
    ///////////////////////////////////



    #define IsActivePrint(_name, color, loginf, logval) \
        if (loglevel >= log_##_name) {logger::raw("    ");} _name##_ln("xxx", "xxx","(","logger::" #_name,")");

    inline void print_active_level(){

        //logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);
        #define X IsActivePrint
        LIST_LEVEL
        #undef X
        //logger::raw_ln(terminal_effects::faint + "----------------------" + terminal_effects::reset);

    }

    #undef IsActivePrint








    
}




