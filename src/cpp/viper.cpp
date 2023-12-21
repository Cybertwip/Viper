#include "viper.hpp"

#include <chrono>
#include <functional>

#include <filesystem>
#include <optional>

#include "piper.hpp"

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

class Viper::CImpl {
public:
	CImpl(std::filesystem::path& dataPath, const std::string& modelBaseName = "data/ald");
	
	void execute(int speakerId, const std::string& text, std::function<void(std::vector<int16_t>)> onDoneCallback);
	
private:
	void setupViper(std::filesystem::path& dataPath, const std::string& modelBaseName);
	void setupVoice();
	
private:
	RunConfig runConfig;
	piper::PiperConfig piperConfig;
	piper::Voice voice;
	piper::SynthesisResult result;
	
	std::vector<int16_t> audioBuffer;
};

Viper::CImpl::CImpl(std::filesystem::path& dataPath, const std::string& modelBaseName) {
	setupViper(dataPath, modelBaseName);
}

void Viper::CImpl::execute(int speakerId, const std::string& text, std::function<void(std::vector<int16_t>)> onDoneCallback){
		
	result = piper::SynthesisResult();
	
	auto defaultSpeakerId = voice.synthesisConfig.speakerId;
	
	voice.synthesisConfig.speakerId = speakerId;
	
	auto audioCallback = [this, defaultSpeakerId, onDoneCallback]() {
		// Signal thread that audio is ready
		{
			if(onDoneCallback){
				onDoneCallback(std::move(audioBuffer));
			}
			
			voice.synthesisConfig.speakerId = defaultSpeakerId;
		}
	};
	
	piper::textToAudio(piperConfig, voice, text, audioBuffer, result, audioCallback);
}
		
void Viper::CImpl::setupViper(std::filesystem::path& dataPath, const std::string& modelBaseName){
	
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

void Viper::CImpl::setupVoice(){
	
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

Viper::Viper(const std::string& dataPath, const std::string& modelBaseName)
{
	auto path = std::filesystem::path(dataPath);
	impl = std::make_unique<CImpl>(path, modelBaseName);
}

Viper::~Viper()
{
}

void Viper::execute(int speakerId, const std::string& text, std::function<void(std::vector<int16_t>)> onDoneCallback){
	impl->execute(speakerId, text, onDoneCallback);
}



