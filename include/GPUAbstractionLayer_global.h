#ifndef GAL_GLOBAL_H
#define GAL_GLOBAL_H

// Generic helper definitions for shared library support
#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(_MSC_VER)   // Windows && MS Visual C
#        if _MSC_VER < 1310    //Version < 7.1?
#            pragma message ("Compiling with a Visual C compiler version < 7.1 (2003) has not been tested!")
#        endif // Version > 7.1
#        define MYLIB_HELPER_DLL_IMPORT __declspec(dllimport)
#        define MYLIB_HELPER_DLL_EXPORT __declspec(dllexport)
#        define MYLIB_HELPER_DLL_LOCAL
#    elif defined (__GNUC__)
#        define MYLIB_HELPER_DLL_IMPORT __attribute__((dllimport))
#        define MYLIB_HELPER_DLL_EXPORT __attribute__((dllexport))
#        define MYLIB_HELPER_DLL_LOCAL
#    endif
#    define BOOST_ALL_NO_LIB  //disable the msvc automatic boost-lib selection in order to link against the static libs!
#elif defined(__linux__) || defined(linux) || defined(__linux)
#    if __GNUC__ >= 4    // TODO Makefile: add -fvisibility=hidden to compiler parameter in Linux version
#        define MYLIB_HELPER_DLL_IMPORT __attribute__ ((visibility("default")))
#        define MYLIB_HELPER_DLL_EXPORT __attribute__ ((visibility("default")))
#        define MYLIB_HELPER_DLL_LOCAL  __attribute__ ((visibility("hidden")))
#    else
#        define MYLIB_HELPER_DLL_IMPORT
#        define MYLIB_HELPER_DLL_EXPORT
#        define MYLIB_HELPER_DLL_LOCAL
#    endif
#endif

// Now we use the generic helper definitions above to define MYLIB_API and MYLIB_LOCAL.
// MYLIB_API is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
// MYLIB_LOCAL is used for non-api symbols.

#ifndef MYLIB_DLL
#define MYLIB_DLL
#endif
// TODO Makefile: add MYLIB_DLL and EXPORTS
#ifdef MYLIB_DLL // defined if MYLIB is compiled as a DLL
#    ifdef GAL_LIBRARY // defined if we are building the MYLIB DLL (instead of using it)
#        define GAL_EXPORT MYLIB_HELPER_DLL_EXPORT
#		 define EXPIMP_TEMPLATE
#    else
#        define GAL_EXPORT MYLIB_HELPER_DLL_IMPORT
#		 define EXPIMP_TEMPLATE extern
#    endif // MYLIB_DLL_EXPORTS
#define DLL_LOCAL MYLIB_HELPER_DLL_LOCAL
#else // MYLIB_DLL is not defined: this means MYLIB is a static lib.
#	 define DLL_PUBLIC
#    define DLL_LOCAL
#endif // MYLIB_DLL

//#if defined(PSE_LIBRARY)
//#  define GAL_EXPORT Q_DECL_EXPORT
//#else
//#  define GAL_EXPORT Q_DECL_IMPORT
//#endif

#endif // PROTOSPHEREEXTENDED_GLOBAL_H
