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


VIPER_DLL class Viper2 {
public:
	Viper2(const std::string& dataPath,
		   const std::string& chatModel,
		   const std::string& modelBaseName);
	~Viper2();
	
	void chat(int speakerTurn, 
			  std::function<void(std::pair<int, std::string>)> textCallback,
			  std::function<void(std::vector<int16_t>)> audioCallback);
	
	bool update();
	
private:
	class CImpl;
	std::unique_ptr<CImpl> impl;
};

