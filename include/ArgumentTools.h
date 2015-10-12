/*
 * ArgumentTools.h
 *
 *  Created on: 25 Aug 2014
 *      Author: jay
 */

#ifndef ARGUMENTTOOLS_H_
#define ARGUMENTTOOLS_H_

#include <string>

#include "GPUAbstractionLayer_global.h"

GAL_EXPORT bool checkArgumentExists( const int argc, const char **argv, std::string strKey );

GAL_EXPORT int getArgumentInt( const int argc, const char **argv, std::string strKey );
GAL_EXPORT float getArgumentFloat( const int argc, const char **argv, std::string strKey );
GAL_EXPORT std::string getArgumentString( const int argc, const char **argv, std::string strKey );

#endif /* ARGUMENTTOOLS_H_ */
