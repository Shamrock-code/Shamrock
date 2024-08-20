#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"

inline std::string
reformat_all(std::string color, const char *name, std::string module_name, std::string content) {
    // old form
    //return "[" + (color) + module_name + shambase::term_colors::reset()
    //       + "] " + (color) + (name)
    //       + shambase::term_colors::reset() + ": " + content;

    // new form
    //shambase::replace_all(content, "\n", "\n                                   | ");
    return shambase::format(
        "{5:}rank={6:<4}{2:} {5:}({3:^20}){2:} {0:}{1:}{2:}: {4:}",
        color,
        name,
        shambase::term_colors::reset(),
        module_name,
        content,
        shambase::term_colors::faint()
        ,shamcomm::world_rank());
}

inline std::string
reformat_simple(std::string color, const char *name, std::string module_name, std::string content) {
    // old form
    //return "[" + (color) + module_name + shambase::term_colors::reset()
    //       + "] " + (color) + (name)
    //       + shambase::term_colors::reset() + ": " + content;

    // new form
    //shambase::replace_all(content, "\n", "\n                                   | ");
    return shambase::format(
        "{5:}({3:}){2:} : {4:}",
        color,
        name,
        shambase::term_colors::reset(),
        module_name,
        content,
        shambase::term_colors::faint());
}

std::string LogLevel_DebugAlloc::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_red(), level_name, module_name, in);
}

std::string LogLevel_DebugMPI::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_blue(), level_name, module_name, in);
}

std::string LogLevel_DebugSYCL::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_magenta(), level_name, module_name, in);
}

std::string LogLevel_Debug::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_green(), level_name, module_name, in);
}

std::string LogLevel_Info::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_cyan(), "Info", module_name, in);
}

std::string LogLevel_Normal::reformat(const std::string &in, std::string module_name) {
    return ::reformat_simple(shambase::term_colors::empty(), level_name, module_name, in);
}

std::string LogLevel_Warning::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_yellow(), level_name, module_name, in);
}

std::string LogLevel_Error::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_red(), level_name, module_name, in);
}
