/**
@brief class for basic system operations
@author: Shane Yuan
@date: Dec 11, 2017
*/

#ifndef __TINY_GIGA_SYS_UTIL_HPP__
#define __TINY_GIGA_SYS_UTIL_HPP__

#include <iostream>
#include <fstream>
#ifdef WIN32
#include <Windows.h>
#endif
#include <direct.h>
#include <chrono>
#include <memory>
#include <thread>

enum class ConsoleColor {
	red = 12,
	blue = 9,
	green = 10,
	yellow = 14,
	white = 15,
	pink = 13,
	cyan = 11
};

#ifndef WIN32
#define BLACK_TEXT(x) "\033[30;1m" x "\033[0m"
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[32;1m" x "\033[0m"
#define YELLOW_TEXT(x) "\033[33;1m" x "\033[0m"
#define BLUE_TEXT(x) "\033[34;1m" x "\033[0m"
#define MAGENTA_TEXT(x) "\033[35;1m" x "\033[0m"
#define CYAN_TEXT(x) "\033[36;1m" x "\033[0m"
#define WHITE_TEXT(x) "\033[37;1m" x "\033[0m"
#endif

class SysUtil {
public:
	/***********************************************************/
	/*                    mkdir function                       */
	/***********************************************************/
	static int mkdir(char* dir) {
#ifdef WIN32
		_mkdir(dir);
#else
		char command[COMMAND_STRING_LENGTH];
		sprintf(command, "mkdir %s", dir);
		system(command);
#endif
		return 0;
	}
	static int mkdir(std::string dir) {
		return mkdir((char *)dir.c_str());
	}

	/***********************************************************/
	/*                    sleep function                       */
	/***********************************************************/
	static int sleep(size_t miliseconds) {
		std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
		return 0;
	}

	/***********************************************************/
	/*             make colorful console output                */
	/***********************************************************/
	static int setConsoleColor(ConsoleColor color) {
#ifdef WIN32
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), static_cast<int>(color));
#endif
		return 0;
	}

	/***********************************************************/
	/*                 warning error output                    */
	/***********************************************************/
	static int errorOutput(std::string info) {
#ifdef WIN32
		SysUtil::setConsoleColor(ConsoleColor::red);
		std::cerr << "ERROR: " << info.c_str() << std::endl;
		SysUtil::setConsoleColor(ConsoleColor::white);
#else
		std::cerr << RED_TEXT("ERROR: ") << RED_TEXT(info.c_str()) << std::endl;
#endif
		return 0;
	}

	static int warningOutput(std::string info) {
#ifdef WIN32
		SysUtil::setConsoleColor(ConsoleColor::yellow);
		std::cerr << "WARNING: " << info.c_str() << std::endl;
		SysUtil::setConsoleColor(ConsoleColor::white);
#else
		std::cerr << YELLOW_TEXT("ERROR: ") << YELLOW_TEXT(info.c_str()) << std::endl;
#endif
		return 0;
	}

	static int infoOutput(std::string info) {
#ifdef WIN32
		SysUtil::setConsoleColor(ConsoleColor::green);
		std::cerr << "INFO: " << info.c_str() << std::endl;
		SysUtil::setConsoleColor(ConsoleColor::white);
#else
		std::cerr << GREEN_TEXT("ERROR: ") << GREEN_TEXT(info.c_str()) << std::endl;
#endif
		return 0;
	}

};


#endif