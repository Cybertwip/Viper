#include <memory>
#include <string>
#include <filesystem>

class Alpaca {
public:
	Alpaca(std::filesystem::path& dataPath, const std::string& chatModel);
	~Alpaca();
	
	void Chat(int interactionTurn, std::function<void(std::vector<std::pair<int, std::string>>)> callback, const std::string& playerName, const std::string& playerDescription, const std::string& npcName, const std::string& npcDescription, const std::string& playerPrompt = "", const std::string& npcPrompt = "", bool interactionEnd = false);
	
	bool Update();
	
private:
	class CImpl;
	std::unique_ptr<CImpl> impl;
};
