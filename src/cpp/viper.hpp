#pragma once 
#include <memory>
#include <functional>
#include <vector>
#include <string>


#if defined(_WINDOWS) && defined(viper_EXPORTS)
#   define VIPER_DLL extern "C" __declspec(dllexport)
#else
#   define VIPER_DLL
#endif


VIPER_DLL class Viper {
public:
	Viper(const std::string& dataPath, const std::string& modelBaseName = "ald");
	~Viper();
	void execute(int speakerId, const std::string& text, std::function<void(std::vector<int16_t>)> onDoneCallback);
	
private:
	class CImpl;
	std::unique_ptr<CImpl> impl;
};

