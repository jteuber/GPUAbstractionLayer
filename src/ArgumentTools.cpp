/*
 * ArgumentTools.cpp
 *
 *  Created on: 25 Aug 2014
 *      Author: jay
 */

#include "ArgumentTools.h"

#include <map>
#include <sstream>

std::map<std::string, std::string> g_argumentMap;

void buildArgumentMap( const int argc, const char **argv )
{
	for( int i = 1; i < argc; i++ )
	{
		if( i+1 < argc && argv[i+1][0] != '-' )
		{
			g_argumentMap.insert( std::make_pair(std::string(argv[i]), std::string(argv[i+1])) );
			++i;
		}
		else
			g_argumentMap.insert( std::make_pair(std::string(argv[i]), "") );
	}
}


bool checkArgumentExists( const int argc, const char **argv, std::string strKey )
{
	if( g_argumentMap.empty() )
		buildArgumentMap( argc, argv );

	return g_argumentMap.find(strKey) != g_argumentMap.end();
}

int getArgumentInt( const int argc, const char **argv, std::string strKey )
{
	if( g_argumentMap.empty() )
		buildArgumentMap( argc, argv );

	int iRet = 0;
	std::stringstream ss;
	ss << g_argumentMap[strKey];
	ss >> iRet;

	return iRet;
}

float getArgumentFloat( const int argc, const char **argv, std::string strKey )
{
	if( g_argumentMap.empty() )
		buildArgumentMap( argc, argv );

	float fRet = 0;
	std::stringstream ss;
	ss << g_argumentMap[strKey];
	ss >> fRet;

	return fRet;
}

std::string getArgumentString( const int argc, const char **argv, std::string strKey )
{
	if( g_argumentMap.empty() )
		buildArgumentMap( argc, argv );

	return g_argumentMap[strKey];
}

