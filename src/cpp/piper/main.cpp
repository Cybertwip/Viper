#include "viper.hpp"
#include <spdlog/spdlog.h>


int main(int argc, char** argv){
	std::string dataPath = "data";

	Viper2 viper(dataPath, "tinyllama-1.1b-chat-v0.6/ggml-model-q4_0.gguf", "sharvard-medium");
	
	bool running = true;
	
	auto audioCallback = [&running](std::vector<int16_t> output){
		running = false;
	};

	auto textCallback = [&running](std::pair<int, std::string> output){
		spdlog::debug("Parsing voice config at {}", output.second);
	};

	viper.chat(1, textCallback, audioCallback);
	
	while(running){
		viper.update();
	}
	
	return 0;
}
