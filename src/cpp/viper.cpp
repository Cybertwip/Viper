#include "viper.hpp"

#include <chrono>
#include <functional>

#include <filesystem>
#include <optional>
#include <vector>

#include "piper.hpp"
#include "alpaca.hpp"

namespace {
void replaceWithinDelimiters(std::string& myString, const std::string& openingDelimiter, const std::string& closingDelimiter, const std::string& replaceString) {
    size_t pos = myString.find(openingDelimiter);

    while (pos != std::string::npos) {
        size_t endPos = myString.find(closingDelimiter, pos + openingDelimiter.length());
        if (endPos != std::string::npos) {
            myString.replace(pos, endPos - pos + closingDelimiter.length(), replaceString);
        } else {
            // Handle the case where the closing delimiter is not found
            break;
        }

        pos = myString.find(openingDelimiter, pos + replaceString.length());
    }
}
void trimSpaces(std::string& myString) {
    // Trim leading spaces
    size_t start = myString.find_first_not_of(" \t\r\n");
    if (start != std::string::npos) {
        myString.erase(0, start);
    } else {
        // If the string is all spaces, clear it
        myString.clear();
        return;
    }

    // Trim trailing spaces
    size_t end = myString.find_last_not_of(" \t\r\n");
    if (end != std::string::npos) {
        myString.erase(end + 1);
    }
}

}

struct RunConfig {
	// Path to .onnx voice file
	std::filesystem::path modelPath;
	
	// Path to JSON voice config file
	std::filesystem::path modelConfigPath;
	
	// Numerical id of the default speaker (multi-speaker voices)
	std::optional<piper::SpeakerId> speakerId;
	
	// Amount of noise to add during audio generation
	std::optional<float> noiseScale;
	
	// Speed of speaking (1 = normal, < 1 is faster, > 1 is slower)
	std::optional<float> lengthScale;
	
	// Variation in phoneme lengths
	std::optional<float> noiseW;
	
	// Seconds of silence to add after each sentence
	std::optional<float> sentenceSilenceSeconds;
	
	// Path to libtashkeel ort model
	// https://github.com/mush42/libtashkeel/
	std::optional<std::filesystem::path> tashkeelModelPath;
	
	// Seconds of extra silence to insert after a single phoneme
	std::optional<std::map<piper::Phoneme, float>> phonemeSilenceSeconds;
};

class Viper2::CImpl {
public:
	CImpl(std::filesystem::path& dataPath,
		  const std::string& chatModel = "data/llama-2-7b-chat/ggml-model-f16.gguf", const std::string& modelBaseName = "data/ald");
	~CImpl();

	void chat(int speakerTurn, std::function<void(std::vector<std::pair<int, std::string>>)> callback);
	
	void save(std::ostream &audioFile, std::vector<int16_t> audioBuffer);
	
	std::vector<int16_t> execute(int speakerId, const std::string& text);
	
	bool update();

private:
	void setupViper(std::filesystem::path& dataPath, const std::string& modelBaseName);
	void setupVoice();
	
private:
	RunConfig runConfig;
	piper::PiperConfig piperConfig;
	piper::Voice voice;
	piper::SynthesisResult result;
	
	std::vector<int16_t> audioBuffer;
	
	std::unique_ptr<Alpaca> alpaca;
};

Viper2::CImpl::~CImpl(){
	
}

Viper2::CImpl::CImpl(std::filesystem::path& dataPath, const std::string& chatModel, const std::string& modelBaseName) : alpaca(std::make_unique<Alpaca>(dataPath, chatModel)) {
	setupViper(dataPath, modelBaseName);
}

void Viper2::CImpl::chat(int speakerTurn, std::function<void(std::vector<std::pair<int, std::string>>)> callback){
	
	alpaca->Chat(speakerTurn, callback, "Señor Patata", "Una patata enojona", "Regan", "Una Pirata");
}

void Viper2::CImpl::save(std::ostream &audioFile, std::vector<int16_t> audioBuffer){
	piper::pcmToWavFile(voice, audioFile, audioBuffer);
}

std::vector<int16_t> Viper2::CImpl::execute(int speakerId, const std::string& text){

	audioBuffer.clear();
	result = piper::SynthesisResult();
	voice.synthesisConfig.speakerId = speakerId;
	piper::textToAudio(piperConfig, voice, text, audioBuffer, result, nullptr);
	return audioBuffer;
}
		
void Viper2::CImpl::setupViper(std::filesystem::path& dataPath, const std::string& modelBaseName){
	
	runConfig.modelPath = std::filesystem::absolute(
						dataPath / (modelBaseName + ".onnx")).string();

	runConfig.modelConfigPath = std::filesystem::absolute(
						dataPath / (modelBaseName + ".onnx.json")).string();
	
	auto startTime = std::chrono::steady_clock::now();
	loadVoice(piperConfig, runConfig.modelPath.string(),
			  runConfig.modelConfigPath.string(), voice, runConfig.speakerId,
			  false);
	auto endTime = std::chrono::steady_clock::now();
		
	piperConfig.eSpeakDataPath =
	std::filesystem::absolute(
							  dataPath / "espeak-ng-data").string();
		
	if (voice.phonemizeConfig.phonemeType != piper::eSpeakPhonemes) {
		// Not using eSpeak
		piperConfig.useESpeak = false;
	}
	
	// Enable libtashkeel for Arabic
	if (voice.phonemizeConfig.eSpeak.voice == "ar") {
		piperConfig.useTashkeel = true;
		
		piperConfig.tashkeelModelPath =
		std::filesystem::absolute(dataPath / "libtashkeel_model.ort").string();
	}
	
	piper::initialize(piperConfig);
}

void Viper2::CImpl::setupVoice(){
	// Scales
	if (runConfig.noiseScale) {
		voice.synthesisConfig.noiseScale = runConfig.noiseScale.value();
	}
	
	if (runConfig.lengthScale) {
		voice.synthesisConfig.lengthScale = runConfig.lengthScale.value();
	}
	
	if (runConfig.noiseW) {
		voice.synthesisConfig.noiseW = runConfig.noiseW.value();
	}
	
	if (runConfig.sentenceSilenceSeconds) {
		voice.synthesisConfig.sentenceSilenceSeconds =
		runConfig.sentenceSilenceSeconds.value();
	}
	
	if (runConfig.phonemeSilenceSeconds) {
		if (!voice.synthesisConfig.phonemeSilenceSeconds) {
			// Overwrite
			voice.synthesisConfig.phonemeSilenceSeconds =
			runConfig.phonemeSilenceSeconds;
		} else {
			// Merge
			for (const auto &[phoneme, silenceSeconds] :
				 *runConfig.phonemeSilenceSeconds) {
				voice.synthesisConfig.phonemeSilenceSeconds->try_emplace(
																		 phoneme, silenceSeconds);
			}
		}
		
	} // if phonemeSilenceSeconds
}

bool Viper2::CImpl::update(){
	return alpaca->Update();
}

Viper2::Viper2(const std::string& dataPath, const std::string& chatModel,
    const std::string& modelBaseName)
{
	auto path = std::filesystem::path(dataPath);
	impl = std::make_unique<CImpl>(path, chatModel, modelBaseName);
}

Viper2::~Viper2()
{
}

bool Viper2::update() {
	return impl->update();
}

void Viper2::chat(int speakerTurn, 
std::function<void(std::pair<int, std::string>)> textCallback,
std::function<void(std::vector<int16_t>)> audioCallback){
	auto chatCallback = [this, textCallback, audioCallback](std::vector<std::pair<int, std::string>> chatOutput){
		
		std::vector<int16_t> pcmOutput;
		
		std::vector<int16_t> pcmSilence = std::vector<int16_t>(8000);
		
		for(auto& chat : chatOutput){

				
			std::string postprocessedText = chat.second;
			
			replaceWithinDelimiters(postprocessedText, "*", "*", "");
			replaceWithinDelimiters(postprocessedText, "(", ")", "");

			trimSpaces(postprocessedText);



			if(textCallback){
				textCallback({chat.first, postprocessedText + "\n"});
			}

			auto pcm = impl->execute(chat.first, postprocessedText);
			
			pcmOutput.insert(pcmOutput.end(), pcm.begin(), pcm.end());

			pcmOutput.insert(pcmOutput.end(), pcmSilence.begin(), pcmSilence.end());

		}

		if(audioCallback){
			audioCallback(pcmOutput);
		}

		// std::string outputPath = "audio.wav";
		// std::ofstream audioFile(outputPath, std::ios::binary);
		// impl->save(audioFile, pcmOutput);
	};
	
	impl->chat(speakerTurn, chatCallback);

}





