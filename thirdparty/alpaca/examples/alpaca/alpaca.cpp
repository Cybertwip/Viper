#include "alpaca.hpp"

#include "common.h"
#include "llama.h"
#include "grammar-parser.h"

#include "../llava/clip.h"

#include "stb_image.h"
#include "json.hpp"

#include <cstddef>
#include <thread>
#include <mutex>
#include <chrono>
#include <set>
#include <memory>
#include <filesystem>

#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
#endif

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

using json = nlohmann::json;

static bool server_verbose = false;

#if SERVER_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                            \
    do                                                                   \
    {                                                                    \
        if (server_verbose)                                              \
        {                                                                \
            server_log("VERBOSE", __func__, __LINE__, MSG, __VA_ARGS__); \
        }                                                                \
    } while (0)
#endif

#define LOG_ERROR(  MSG, ...) server_log("ERROR",   __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log("WARNING", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_INFO(   MSG, ...) server_log("INFO",    __func__, __LINE__, MSG, __VA_ARGS__)

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static std::vector<uint8_t> base64_decode(std::string const &encoded_string)
{
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_]))
    {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4)
        {
            for (i = 0; i <4; i++)
            {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++)
            {
                ret.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i)
    {
        for (j = i; j <4; j++)
        {
            char_array_4[j] = 0;
        }

        for (j = 0; j <4; j++)
        {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; (j < i - 1); j++)
        {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// parallel
//

enum task_type {
    COMPLETION_TASK,
    CANCEL_TASK
};

struct task_server {
    int id;
    int target_id;
    task_type type;
    json data;
    bool infill_mode = false;
    bool embedding_mode = false;
    int multitask_id = -1;
};

struct task_result {
    int id;
    int multitask_id = -1;
    bool stop;
    bool error;
    json result_json;
};

struct task_multi {
    int id;
    std::set<int> subtasks_remaining{};
    std::vector<task_result> results{};
};

// TODO: can become bool if we can't find use of more states
enum slot_state
{
    IDLE,
    PROCESSING,
};

enum slot_command
{
    NONE,
    LOAD_PROMPT,
    RELEASE,
};

struct slot_params
{
    bool stream       = true;
    bool cache_prompt = false; // remember the prompt to avoid reprocessing all prompt

    uint32_t seed      = -1; // RNG seed
    int32_t  n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t  n_predict = -1; // new tokens to predict

    std::vector<std::string> antiprompt;

    json input_prefix;
    json input_suffix;
};

struct slot_image
{
    int32_t id;

    bool request_encode_image = false;
    float* image_embedding = nullptr;
    int32_t image_tokens = 0;

    clip_image_u8 img_data;

    std::string prefix_prompt; // before of this image
};

// completion token output with probabilities
struct completion_token_output
{
    struct token_prob
    {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
    llama_token tok;
    std::string text_to_send;
};

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }
    return i;
}

enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

static bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text)
{
    if (!text.empty() && !stop.empty())
    {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--)
        {
            if (stop[char_index] == text_last_char)
            {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial))
                {
                    return text.size() - char_index - 1;
                }
            }
        }
    }
    return std::string::npos;
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context *ctx, Iter begin, Iter end)
{
    std::string ret;
    for (; begin != end; ++begin)
    {
        ret += llama_token_to_piece(ctx, *begin);
    }
    return ret;
}

static void server_log(const char *level, const char *function, int line,
                       const char *message, const nlohmann::ordered_json &extra)
{
    nlohmann::ordered_json log
    {
        {"timestamp", time(nullptr)},
        {"level",     level},
        {"function",  function},
        {"line",      line},
        {"message",   message},
    };

    if (!extra.empty())
    {
        log.merge_patch(extra);
    }

    const std::string str = log.dump(-1, ' ', false, json::error_handler_t::replace);
    printf("%.*s\n", (int)str.size(), str.data());
    fflush(stdout);
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);
    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }
    return out;
}

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context *ctx, const std::vector<completion_token_output> &probs)
{
    json out = json::array();
    for (const auto &prob : probs)
    {
        json probs_for_token = json::array();
        for (const auto &p : prob.probs)
        {
            std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
            probs_for_token.push_back(json
            {
                {"tok_str", tok_str},
                {"prob",    p.prob},
            });
        }
        std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
        out.push_back(json{
            {"content", tok_str},
            {"probs",   probs_for_token},
        });
    }
    return out;
}

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value)
{
    // Fallback null to default value
    return body.contains(key) && !body.at(key).is_null()
        ? body.value(key, default_value)
        : default_value;
}

struct llama_client_slot
{
    int id;
    int task_id = -1;

    struct slot_params params;

    slot_state state = IDLE;
    slot_command command = NONE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_past      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t num_prompt_tokens           = 0;
    int32_t num_prompt_tokens_processed = 0;

    json prompt;
    std::string generated_text;
    llama_token sampled;
    std::vector<llama_token> cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    bool infill = false;
    bool embedding = false;
    bool has_next_token = true;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;

    bool oaicompat = false;
    std::string oaicompat_model;

    std::string stopping_word;

    // sampling
    struct llama_sampling_params sparams;
    llama_sampling_context *ctx_sampling = nullptr;

    // multimodal
    std::vector<slot_image> images;

    // stats
    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_genereration;

    double t_prompt_processing; // ms
    double t_token_generation; // ms

    // multitasks
    int multitask_id = -1;

    void reset() {
        num_prompt_tokens      = 0;
        generated_text         = "";
        truncated              = false;
        stopped_eos            = false;
        stopped_word           = false;
        stopped_limit          = false;
        stopping_word          = "";
        n_past                 = 0;
        sent_count             = 0;
        sent_token_probs_index = 0;
        infill                 = false;

        generated_token_probs.clear();

        for (slot_image &img : images)
        {
            free(img.image_embedding);
            delete[] img.img_data.data;
            img.prefix_prompt = "";
        }

        images.clear();
        // llama_set_rng_seed(ctx, params.seed); in batched the seed matter???????
    }

    bool has_budget(gpt_params &global_params) {
        n_remaining = -1;
        if(params.n_predict != -1)
        {
            n_remaining = params.n_predict - n_decoded;
        }
        else if (global_params.n_predict != -1)
        {
            n_remaining = global_params.n_predict - n_decoded;
        }
        return n_remaining > 0 || n_remaining == -1; // no budget || limitless
    }

    bool available() const {
        return state == IDLE && command == NONE;
    }

    bool is_processing() const {
        return (state == IDLE && command == LOAD_PROMPT) || state == PROCESSING;
    }

    void add_token_string(const completion_token_output &token) {
        if (command == RELEASE)
        {
            return;
        }
        cache_tokens.push_back(token.tok);
        generated_token_probs.push_back(token);
    }

    void release() {
        if (state == IDLE || state == PROCESSING)
        {
            t_token_generation = (ggml_time_us() - t_start_genereration) / 1e3;
            command = RELEASE;
        }
    }

    json get_formated_timings() {
        return json
        {
            {"prompt_n",               num_prompt_tokens_processed},
            {"prompt_ms",              t_prompt_processing},
            {"prompt_per_token_ms",    t_prompt_processing / num_prompt_tokens_processed},
            {"prompt_per_second",      1e3 / t_prompt_processing * num_prompt_tokens_processed},

            {"predicted_n",            n_decoded},
            {"predicted_ms",           t_token_generation},
            {"predicted_per_token_ms", t_token_generation / n_decoded},
            {"predicted_per_second",   1e3 / t_token_generation * n_decoded},
        };
    }

    void print_timings() const {
        LOG_TEE("\n");
        LOG_TEE("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, t_prompt_processing, num_prompt_tokens_processed, t_prompt_processing / num_prompt_tokens_processed, 1e3 / t_prompt_processing * num_prompt_tokens_processed);
        LOG_TEE("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, t_token_generation, n_decoded,t_token_generation / n_decoded, 1e3 / t_token_generation * n_decoded);
        LOG_TEE("%s:       total time = %10.2f ms\n", __func__, t_prompt_processing + t_token_generation);
    }
};

struct llama_server_context
{
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;

    clip_ctx *clp_ctx = nullptr;

    gpt_params params;

    llama_batch batch;

    bool multimodal         = false;
    bool clean_kv_cache     = true;
    bool all_slots_are_idle = false;
    bool add_bos_token      = true;

    int32_t id_gen;
    int32_t n_ctx;  // total context for all clients / slots

    // system prompt
    bool system_need_update = false;

    std::string              system_prompt;
    std::vector<llama_token> system_tokens;

    std::string name_user;      // this should be the antiprompt
    std::string name_assistant;

    // slots / clients
    std::vector<llama_client_slot> slots;

    std::vector<task_server> queue_tasks;
    std::vector<task_result> queue_results;
    std::vector<task_multi>  queue_multitasks;
    std::mutex mutex_tasks; // also guards id_gen, and queue_multitasks
    std::mutex mutex_results;

    ~llama_server_context()
    {
        if (ctx)
        {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model)
        {
            llama_free_model(model);
            model = nullptr;
        }
    }

    bool load_model(const gpt_params &params_)
    {
        params = params_;
        if (!params.mmproj.empty()) {
            multimodal = true;
            LOG_TEE("Multi Modal Mode Enabled");
            clp_ctx = clip_model_load(params.mmproj.c_str(), /*verbosity=*/ 1);
            if(clp_ctx == nullptr) {
                LOG_ERROR("unable to load clip model", {{"model", params.mmproj}});
                return false;
            }

            if (params.n_ctx < 2048) { // request larger context for the image embedding
                params.n_ctx = 2048;
            }
        }

        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        if (model == nullptr)
        {
            LOG_ERROR("unable to load model", {{"model", params.model}});
            return false;
        }

        if (multimodal) {
            const int n_embd_clip = clip_n_mmproj_embd(clp_ctx);
            const int n_embd_llm  = llama_n_embd(model);
            if (n_embd_clip != n_embd_llm) {
                LOG_TEE("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_embd_clip, n_embd_llm);
                llama_free(ctx);
                llama_free_model(model);
                return false;
            }
        }

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_should_add_bos_token(model);

        return true;
    }

    void initialize() {
        id_gen = 0;

        // create slots
        all_slots_are_idle = true;

        const int32_t n_ctx_slot = n_ctx / params.n_parallel;

        LOG_TEE("Available slots:\n");
        for (int i = 0; i < params.n_parallel; i++)
        {
            llama_client_slot slot;

            slot.id = i;
            slot.n_ctx = n_ctx_slot;
            slot.reset();

            LOG_TEE(" -> Slot %i - max context: %i\n", slot.id, n_ctx_slot);
            slots.push_back(slot);
        }

        batch = llama_batch_init(n_ctx, 0, params.n_parallel);

        // empty system prompt
        system_prompt = "";
        system_tokens.clear();
    }

    std::vector<llama_token> tokenize(const json & json_prompt, bool add_bos) const
    {
        // TODO: currently, we tokenize using special tokens by default
        //       this is not always correct (see https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824826216)
        //       but it's better compared to completely ignoring ChatML and other chat templates
        const bool TMP_FORCE_SPECIAL = true;

        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        std::vector<llama_token> prompt_tokens;

        if (json_prompt.is_array())
        {
            bool first = true;
            for (const auto& p : json_prompt)
            {
                if (p.is_string())
                {
                    auto s = p.template get<std::string>();
                    std::vector<llama_token> p;
                    if (first)
                    {
                        p = ::llama_tokenize(ctx, s, add_bos, TMP_FORCE_SPECIAL);
                        first = false;
                    }
                    else
                    {
                        p = ::llama_tokenize(ctx, s, false, TMP_FORCE_SPECIAL);
                    }
                    prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
                }
                else
                {
                    if (first)
                    {
                        first = false;
                    }
                    prompt_tokens.push_back(p.template get<llama_token>());
                }
            }
        }
        else
        {
            auto s = json_prompt.template get<std::string>();
            prompt_tokens = ::llama_tokenize(ctx, s, add_bos, TMP_FORCE_SPECIAL);
        }

        return prompt_tokens;
    }

    llama_client_slot* get_slot(int id) {
        int64_t t_last = ggml_time_us();
        llama_client_slot *last_used = nullptr;

        for (llama_client_slot & slot : slots)
        {
            if (slot.id == id && slot.available())
            {
                return &slot;
            }

            if (slot.available() && slot.t_last_used < t_last)
            {
                last_used = &slot;
                t_last = slot.t_last_used;
            }
        }

        return last_used;
    }

    bool launch_slot_with_data(llama_client_slot* &slot, json data) {
        slot_params default_params;
        llama_sampling_params default_sparams;

        if (data.count("__oaicompat") != 0) {
            slot->oaicompat = true;
            slot->oaicompat_model = json_value(data, "model", std::string(DEFAULT_OAICOMPAT_MODEL));
        } else {
            slot->oaicompat = false;
            slot->oaicompat_model = "";
        }

        slot->params.stream           = json_value(data, "stream",            false);
        slot->params.cache_prompt     = json_value(data, "cache_prompt",      false);
        slot->params.n_predict        = json_value(data, "n_predict",         default_params.n_predict);
        slot->sparams.top_k           = json_value(data, "top_k",             default_sparams.top_k);
        slot->sparams.top_p           = json_value(data, "top_p",             default_sparams.top_p);
        slot->sparams.min_p           = json_value(data, "min_p",             default_sparams.min_p);
        slot->sparams.tfs_z           = json_value(data, "tfs_z",             default_sparams.tfs_z);
        slot->sparams.typical_p       = json_value(data, "typical_p",         default_sparams.typical_p);
        slot->sparams.temp            = json_value(data, "temperature",       default_sparams.temp);
        slot->sparams.penalty_last_n  = json_value(data, "repeat_last_n",     default_sparams.penalty_last_n);
        slot->sparams.penalty_repeat  = json_value(data, "repeat_penalty",    default_sparams.penalty_repeat);
        slot->sparams.penalty_freq    = json_value(data, "frequency_penalty", default_sparams.penalty_freq);
        slot->sparams.penalty_present = json_value(data, "presence_penalty",  default_sparams.penalty_present);
        slot->sparams.mirostat        = json_value(data, "mirostat",          default_sparams.mirostat);
        slot->sparams.mirostat_tau    = json_value(data, "mirostat_tau",      default_sparams.mirostat_tau);
        slot->sparams.mirostat_eta    = json_value(data, "mirostat_eta",      default_sparams.mirostat_eta);
        slot->sparams.penalize_nl     = json_value(data, "penalize_nl",       default_sparams.penalize_nl);
        slot->params.n_keep           = json_value(data, "n_keep",            slot->params.n_keep);
        slot->params.seed             = json_value(data, "seed",              default_params.seed);
        slot->sparams.grammar         = json_value(data, "grammar",           default_sparams.grammar);
        slot->sparams.n_probs         = json_value(data, "n_probs",           default_sparams.n_probs);

        // infill
        if (data.count("input_prefix") != 0)
        {
            slot->params.input_prefix = data["input_prefix"];
        }
        else
        {
            slot->params.input_prefix = "";
        }

        if (data.count("input_suffix") != 0)
        {
            slot->params.input_suffix = data["input_suffix"];
        }
        else
        {
            slot->params.input_suffix = "";
        }

        if (data.count("prompt") != 0)
        {
            slot->prompt = data["prompt"];
        }
        else
        {
            slot->prompt = "";
        }

        slot->sparams.logit_bias.clear();

        if (json_value(data, "ignore_eos", false))
        {
            slot->sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
        }

        const auto &logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && logit_bias->is_array())
        {
            const int n_vocab = llama_n_vocab(model);
            for (const auto &el : *logit_bias)
            {
                if (el.is_array() && el.size() == 2 && el[0].is_number_integer())
                {
                    llama_token tok = el[0].get<llama_token>();
                    if (tok >= 0 && tok < n_vocab)
                    {
                        if (el[1].is_number())
                        {
                            slot->sparams.logit_bias[tok] = el[1].get<float>();
                        }
                        else if (el[1].is_boolean() && !el[1].get<bool>())
                        {
                            slot->sparams.logit_bias[tok] = -INFINITY;
                        }
                    }
                }
            }
        }

        slot->params.antiprompt.clear();

        const auto &stop = data.find("stop");
        if (stop != data.end() && stop->is_array())
        {
            for (const auto &word : *stop)
            {
                if (!word.empty())
                {
                    slot->params.antiprompt.push_back(word);
                }
            }
        }

        if (multimodal)
        {
            const auto &images_data = data.find("image_data");
            if (images_data != data.end() && images_data->is_array())
            {
                for (const auto &img : *images_data)
                {
                    std::string data_b64 = img["data"].get<std::string>();
                    slot_image img_sl;
                    img_sl.id = img.count("id") != 0 ? img["id"].get<int>() : slot->images.size();
                    int width, height, channels;
                    std::vector<uint8_t> image_buffer = base64_decode(data_b64);
                    data_b64.clear();
                    auto data = stbi_load_from_memory(image_buffer.data(), image_buffer.size(), &width, &height, &channels, 3);
                    if (!data) {
                        LOG_TEE("slot %i - failed to load image [id: %i]\n", slot->id, img_sl.id);
                        return false;
                    }
                    LOG_TEE("slot %i - image loaded [id: %i] resolution (%i x %i)\n", slot->id, img_sl.id, width, height);
                    img_sl.img_data.nx = width;
                    img_sl.img_data.ny = height;
                    img_sl.img_data.size = width * height * 3;
                    img_sl.img_data.data = new uint8_t[width * height * 3]();
                    memcpy(img_sl.img_data.data, data, width * height * 3);
                    stbi_image_free(data);
                    img_sl.request_encode_image = true;
                    slot->images.push_back(img_sl);
                }
                // process prompt
                // example: system prompt [img-102] user [img-103] describe [img-134] -> [{id: 102, prefix: 'system prompt '}, {id: 103, prefix: ' user '}, {id: 134, prefix: ' describe '}]}
                if (slot->images.size() > 0 && !slot->prompt.is_array())
                {
                    std::string prompt = slot->prompt.get<std::string>();
                    size_t pos = 0, begin_prefix = 0;
                    std::string pattern = "[img-";
                    while ((pos = prompt.find(pattern, pos)) != std::string::npos) {
                        size_t end_prefix = pos;
                        pos += pattern.length();
                        size_t end_pos = prompt.find("]", pos);
                        if (end_pos != std::string::npos)
                        {
                            std::string image_id = prompt.substr(pos, end_pos - pos);
                            try
                            {
                                int img_id = std::stoi(image_id);
                                bool found = false;
                                for (slot_image &img : slot->images)
                                {
                                    if (img.id == img_id) {
                                        found = true;
                                        img.prefix_prompt = prompt.substr(begin_prefix, end_prefix - begin_prefix);
                                        begin_prefix = end_pos + 1;
                                        break;
                                    }
                                }
                                if (!found) {
                                    LOG_TEE("ERROR: Image with id: %i, not found.\n", img_id);
                                    slot->images.clear();
                                    return false;
                                }
                            } catch (const std::invalid_argument& e) {
                                LOG_TEE("Invalid image number id in prompt\n");
                                slot->images.clear();
                                return false;
                            }
                        }
                    }
                    slot->prompt = "";
                    slot->params.input_suffix = prompt.substr(begin_prefix);
                    slot->params.cache_prompt = false; // multimodal doesn't support cache prompt
                }
            }
        }

        if (slot->ctx_sampling != nullptr)
        {
            llama_sampling_free(slot->ctx_sampling);
        }
        slot->ctx_sampling = llama_sampling_init(slot->sparams);
        slot->command = LOAD_PROMPT;

        all_slots_are_idle = false;

        LOG_TEE("slot %i is processing [task id: %i]\n", slot->id, slot->task_id);

        return true;
    }

    void kv_cache_clear() {
        // clear the entire KV cache
        llama_kv_cache_clear(ctx);
        clean_kv_cache = false;
    }

    void update_system_prompt() {
        system_tokens = ::llama_tokenize(ctx, system_prompt, add_bos_token);

        llama_batch_clear(batch);

        kv_cache_clear();

        for (int i = 0; i < (int) system_tokens.size(); ++i)
        {
            llama_batch_add(batch, system_tokens[i], i, { 0 }, false);
        }

        if (llama_decode(ctx, batch) != 0)
        {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return;
        }

        // assign the system KV cache to all parallel sequences
        for (int32_t i = 1; i < params.n_parallel; ++i)
        {
            llama_kv_cache_seq_cp(ctx, 0, i, 0, system_tokens.size());
        }

        LOG_TEE("system prompt updated\n");
        system_need_update = false;
    }

    void notify_system_prompt_changed() {
        // release all slots
        for (llama_client_slot &slot : slots)
        {
            slot.release();
        }

        system_need_update = true;
    }

    void process_system_prompt_data(const json &sys_props) {
        system_prompt  = sys_props.value("prompt", "");
        name_user      = sys_props.value("anti_prompt", "");
        name_assistant = sys_props.value("assistant_name", "");

        if (slots.size() > 0)
        {
            notify_system_prompt_changed();
        }
    }

    static size_t find_stopping_strings(const std::string &text, const size_t last_token_size,
                                        const stop_type type, llama_client_slot &slot)
    {
        size_t stop_pos = std::string::npos;

        for (const std::string &word : slot.params.antiprompt)
        {
            size_t pos;
            if (type == STOP_FULL)
            {
                const size_t tmp = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
                pos = text.find(word, from_pos);
            }
            else
            {
                pos = find_partial_stop_string(word, text);
            }
            if (pos != std::string::npos &&
                (stop_pos == std::string::npos || pos < stop_pos))
            {
                if (type == STOP_FULL)
                {
                    slot.stopped_word = true;
                    slot.stopping_word = word;
                    slot.has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    bool process_token(completion_token_output &result, llama_client_slot &slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = llama_token_to_piece(ctx, result.tok);
        slot.sampled = result.tok;

        // search stop word and delete it
        slot.generated_text += token_str;
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = false;
        for (unsigned i = 1; i < 5 && i <= slot.generated_text.size(); ++i)
        {
            unsigned char c = slot.generated_text[slot.generated_text.size() - i];
            if ((c & 0xC0) == 0x80)
            {
                // continuation byte: 10xxxxxx
                continue;
            }
            if ((c & 0xE0) == 0xC0)
            {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            }
            else if ((c & 0xF0) == 0xE0)
            {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            }
            else if ((c & 0xF8) == 0xF0)
            {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }

        if (!incomplete)
        {
            size_t pos = std::min(slot.sent_count, slot.generated_text.size());
            const std::string str_test = slot.generated_text.substr(pos);
            bool is_stop_full = false;
            size_t stop_pos = find_stopping_strings(str_test, token_str.size(), STOP_FULL, slot);
            if (stop_pos != std::string::npos)
            {
                is_stop_full = true;
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.sent_count, slot.generated_text.size());
            }
            else
            {
                is_stop_full = false;
                stop_pos = find_stopping_strings(str_test, token_str.size(), STOP_PARTIAL, slot);
            }

            // check if there is any token to predict
            if (stop_pos == std::string::npos || (!slot.has_next_token && !is_stop_full && stop_pos > 0))
            {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.sent_count += result.text_to_send.size();
                // add the token to slot queue and cache
            }
            slot.add_token_string(result);
            if (slot.params.stream)
            {
                send_partial_response(slot, result);
            }
        }

        if (incomplete)
        {
            slot.has_next_token = true;
        }

        // check the limits
        if (slot.n_decoded > 2 && slot.has_next_token && !slot.has_budget(params))
        {
            slot.stopped_limit = true;
            slot.has_next_token = false;
        }

        if (!slot.cache_tokens.empty() && result.tok == llama_token_eos(model))
        {
            slot.stopped_eos = true;
            slot.has_next_token = false;
            LOG_VERBOSE("eos token found", {});
        }

        LOG_VERBOSE("next token", {
                                      {"token", result.tok},
                                      {"token_text", tokens_to_output_formatted_string(ctx, result.tok)},
                                      {"has_next_token", slot.has_next_token},
                                      {"n_remain", slot.n_remaining},
                                      {"num_tokens_predicted", slot.n_decoded},
                                      {"stopped_eos", slot.stopped_eos},
                                      {"stopped_word", slot.stopped_word},
                                      {"stopped_limit", slot.stopped_limit},
                                      {"stopping_word", slot.stopping_word},
                                  });

        return slot.has_next_token; // continue
    }

    bool process_images(llama_client_slot &slot) const
    {
        for (slot_image &img : slot.images)
        {
            if (!img.request_encode_image)
            {
                continue;
            }
            clip_image_f32 img_res;
            if (!clip_image_preprocess(clp_ctx, &img.img_data, &img_res, /*pad2square =*/ true))
            {
                LOG_TEE("Error processing the given image");
                clip_free(clp_ctx);
                return false;
            }
            img.image_tokens = clip_n_patches(clp_ctx);
            img.image_embedding = (float *)malloc(clip_embd_nbytes(clp_ctx));
            if (!img.image_embedding)
            {
                LOG_TEE("Unable to allocate memory for image embeddings\n");
                clip_free(clp_ctx);
                return false;
            }
            LOG_TEE("slot %i - encoding image [id: %i]\n", slot.id, img.id);
            if (!clip_image_encode(clp_ctx, params.n_threads, &img_res, img.image_embedding))
            {
                LOG_TEE("Unable to encode image\n");
                return false;
            }
            img.request_encode_image = false;
        }

        return slot.images.size() > 0;
    }

    void send_error(task_server& task, std::string error)
    {
        std::lock_guard<std::mutex> lock(mutex_results);
        task_result res;
        res.id = task.id;
        res.multitask_id = task.multitask_id;
        res.stop = false;
        res.error = true;
        res.result_json = { { "content", error } };
        queue_results.push_back(res);
    }

    void add_multi_task(int id, std::vector<int>& sub_ids)
    {
        std::lock_guard<std::mutex> lock(mutex_tasks);
        task_multi multi;
        multi.id = id;
        std::copy(sub_ids.begin(), sub_ids.end(), std::inserter(multi.subtasks_remaining, multi.subtasks_remaining.end()));
        queue_multitasks.push_back(multi);
    }

    void update_multi_task(int multitask_id, int subtask_id, task_result& result)
    {
        std::lock_guard<std::mutex> lock(mutex_tasks);
        for (auto& multitask : queue_multitasks)
        {
            if (multitask.id == multitask_id)
            {
                multitask.subtasks_remaining.erase(subtask_id);
                multitask.results.push_back(result);
            }
        }
    }

    json get_model_props()
    {
        return get_formated_generation(slots[0]);
    }

    json get_formated_generation(llama_client_slot &slot)
    {
        const auto eos_bias = slot.sparams.logit_bias.find(llama_token_eos(model));
        const bool ignore_eos = eos_bias != slot.sparams.logit_bias.end() &&
                                eos_bias->second < 0.0f && std::isinf(eos_bias->second);
        return json {
            {"n_ctx",             slot.n_ctx},
            {"model",             params.model_alias},
            {"seed",              slot.params.seed},
            {"temp",              slot.sparams.temp},
            {"top_k",             slot.sparams.top_k},
            {"top_p",             slot.sparams.top_p},
            {"min_p",             slot.sparams.min_p},
            {"tfs_z",             slot.sparams.tfs_z},
            {"typical_p",         slot.sparams.typical_p},
            {"repeat_last_n",     slot.sparams.penalty_last_n},
            {"repeat_penalty",    slot.sparams.penalty_repeat},
            {"presence_penalty",  slot.sparams.penalty_present},
            {"frequency_penalty", slot.sparams.penalty_freq},
            {"mirostat",          slot.sparams.mirostat},
            {"mirostat_tau",      slot.sparams.mirostat_tau},
            {"mirostat_eta",      slot.sparams.mirostat_eta},
            {"penalize_nl",       slot.sparams.penalize_nl},
            {"stop",              slot.params.antiprompt},
            {"n_predict",         slot.params.n_predict},
            {"n_keep",            params.n_keep},
            {"ignore_eos",        ignore_eos},
            {"stream",            slot.params.stream},
            {"logit_bias",        slot.sparams.logit_bias},
            {"n_probs",           slot.sparams.n_probs},
            {"grammar",           slot.sparams.grammar},
        };
    }

    void send_partial_response(llama_client_slot &slot, completion_token_output tkn)
    {
        std::lock_guard<std::mutex> lock(mutex_results);
        task_result res;
        res.id = slot.task_id;
        res.multitask_id = slot.multitask_id;
        res.error = false;
        res.stop = false;

        res.result_json = json
        {
            {"content",    tkn.text_to_send},
            {"stop",       false},
            {"slot_id",    slot.id},
            {"multimodal", multimodal}
        };

        if (slot.sparams.n_probs > 0)
        {
            std::vector<completion_token_output> probs_output = {};
            const std::vector<llama_token> to_send_toks = llama_tokenize(ctx, tkn.text_to_send, false);
            size_t probs_pos = std::min(slot.sent_token_probs_index, slot.generated_token_probs.size());
            size_t probs_stop_pos = std::min(slot.sent_token_probs_index + to_send_toks.size(), slot.generated_token_probs.size());
            if (probs_pos < probs_stop_pos)
            {
                probs_output = std::vector<completion_token_output>(slot.generated_token_probs.begin() + probs_pos, slot.generated_token_probs.begin() + probs_stop_pos);
            }
            slot.sent_token_probs_index = probs_stop_pos;
            res.result_json["completion_probabilities"] = probs_vector_to_json(ctx, probs_output);
        }

        if (slot.oaicompat)
        {
            res.result_json["oaicompat_token_ctr"] = slot.n_decoded;
            res.result_json["model"] = slot.oaicompat_model;
        }

        queue_results.push_back(res);
    }

    void send_final_response(llama_client_slot &slot)
    {
        std::lock_guard<std::mutex> lock(mutex_results);
        task_result res;
        res.id = slot.task_id;
        res.multitask_id = slot.multitask_id;
        res.error = false;
        res.stop = true;

        res.result_json = json
        {
            {"content",             !slot.params.stream ? slot.generated_text : ""},
            {"slot_id",             slot.id},
            {"stop",                true},
            {"model",               params.model_alias},
            {"tokens_predicted",    slot.n_decoded},
            {"tokens_evaluated",    slot.num_prompt_tokens},
            {"generation_settings", get_formated_generation(slot)},
            {"prompt",              slot.prompt},
            {"truncated",           slot.truncated},
            {"stopped_eos",         slot.stopped_eos},
            {"stopped_word",        slot.stopped_word},
            {"stopped_limit",       slot.stopped_limit},
            {"stopping_word",       slot.stopping_word},
            {"tokens_cached",       slot.n_past},
            {"timings",             slot.get_formated_timings()}
        };

        if (slot.sparams.n_probs > 0)
        {
            std::vector<completion_token_output> probs = {};
            if (!slot.params.stream && slot.stopped_word)
            {
                const std::vector<llama_token> stop_word_toks = llama_tokenize(ctx, slot.stopping_word, false);
                probs = std::vector<completion_token_output>(slot.generated_token_probs.begin(), slot.generated_token_probs.end() - stop_word_toks.size());
            }
            else
            {
                probs = std::vector<completion_token_output>(
                                    slot.generated_token_probs.begin(),
                                    slot.generated_token_probs.begin() + slot.sent_token_probs_index);
            }
            res.result_json["completion_probabilities"] = probs_vector_to_json(ctx, probs);
        }

        if (slot.oaicompat)
        {
            res.result_json["oaicompat_token_ctr"] = slot.n_decoded;
            res.result_json["model"] = slot.oaicompat_model;
        }

        // parent multitask, if any, needs to be updated
        if (slot.multitask_id != -1)
        {
            update_multi_task(slot.multitask_id, slot.task_id, res);
        }

        queue_results.push_back(res);
    }

    void send_embedding(llama_client_slot &slot)
    {
        std::lock_guard<std::mutex> lock(mutex_results);
        task_result res;
        res.id = slot.task_id;
        res.multitask_id = slot.multitask_id;
        res.error = false;
        res.stop = true;

        const int n_embd = llama_n_embd(model);
        if (!params.embedding)
        {
            LOG_WARNING("embedding disabled", {
                                                  {"params.embedding", params.embedding},
                                              });
            res.result_json = json
            {
                {"embedding", std::vector<float>(n_embd, 0.0f)},
            };
        }
        else
        {
            const float *data = llama_get_embeddings(ctx);
            std::vector<float> embedding(data, data + n_embd);
            res.result_json = json
            {
                {"embedding", embedding },
            };
        }
        queue_results.push_back(res);
    }

    int request_completion(json data, bool infill, bool embedding, int multitask_id)
    {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        task_server task;
        task.id = id_gen++;
        task.target_id = 0;
        task.data = std::move(data);
        task.infill_mode = infill;
        task.embedding_mode = embedding;
        task.type = COMPLETION_TASK;
        task.multitask_id = multitask_id;

        // when a completion task's prompt array is not a singleton, we split it into multiple requests
        if (task.data.at("prompt").size() > 1)
        {
            lock.unlock(); // entering new func scope
            return split_multiprompt_task(task);
        }

        // otherwise, it's a single-prompt task, we actually queue it
        queue_tasks.push_back(task);
        return task.id;
    }

    task_result next_result(int task_id)
    {
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(5));
            std::lock_guard<std::mutex> lock(mutex_results);

            if (queue_results.empty())
            {
                continue;
            }

            for (int i = 0; i < (int) queue_results.size(); i++)
            {
                // for now, tasks that have associated parent multitasks just get erased once multitask picks up the result
                if (queue_results[i].multitask_id == task_id)
                {
                    update_multi_task(task_id, queue_results[i].id, queue_results[i]);
                    queue_results.erase(queue_results.begin() + i);
                    continue;
                }

                if (queue_results[i].id == task_id)
                {
                    assert(queue_results[i].multitask_id == -1);
                    task_result res = queue_results[i];
                    queue_results.erase(queue_results.begin() + i);
                    return res;
                }
            }
        }

        // never reached
        //return task_result{-1, false, false, {}};
    }

    // for multiple images processing
    bool ingest_images(llama_client_slot &slot, int n_batch)
    {
        int image_idx = 0;

        while (image_idx < (int) slot.images.size())
        {
            slot_image &img = slot.images[image_idx];

            // process prefix prompt
            for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch)
            {
                const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));
                llama_batch batch_view = {
                    n_tokens,
                    batch.token    + i,
                    nullptr,
                    batch.pos      + i,
                    batch.n_seq_id + i,
                    batch.seq_id   + i,
                    batch.logits   + i,
                    0, 0, 0, // unused
                };
                if (llama_decode(ctx, batch_view))
                {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return false;
                }
            }

            // process image with llm
            for (int i = 0; i < img.image_tokens; i += n_batch)
            {
                int n_eval = img.image_tokens - i;
                if (n_eval > n_batch)
                {
                    n_eval = n_batch;
                }

                const int n_embd = llama_n_embd(model);
                llama_batch batch_img = { n_eval, nullptr, (img.image_embedding + i * n_embd), nullptr, nullptr, nullptr, nullptr, slot.n_past, 1, 0, };
                if (llama_decode(ctx, batch_img))
                {
                    LOG_TEE("%s : failed to eval image\n", __func__);
                    return false;
                }
                slot.n_past += n_eval;
            }
            image_idx++;

            llama_batch_clear(batch);

            // append prefix of next image
            const auto json_prompt = (image_idx >= (int) slot.images.size()) ?
                slot.params.input_suffix : // no more images, then process suffix prompt
                (json)(slot.images[image_idx].prefix_prompt);

            std::vector<llama_token> append_tokens = tokenize(json_prompt, false); // has next image
            for (int i = 0; i < (int) append_tokens.size(); ++i)
            {
                llama_batch_add(batch, append_tokens[i], slot.n_past, { slot.id }, true);
                slot.n_past += 1;
            }
        }

        return true;
    }

    void request_cancel(int task_id)
    {
        std::lock_guard<std::mutex> lock(mutex_tasks);
        task_server task;
        task.id = id_gen++;
        task.type = CANCEL_TASK;
        task.target_id = task_id;
        queue_tasks.push_back(task);
    }

    int split_multiprompt_task(task_server& multiprompt_task)
    {
        int prompt_count = multiprompt_task.data.at("prompt").size();
        assert(prompt_count > 1);

        int multitask_id = id_gen++;
        std::vector<int> subtask_ids(prompt_count);
        for (int i = 0; i < prompt_count; i++)
        {
            json subtask_data = multiprompt_task.data;
            subtask_data["prompt"] = subtask_data["prompt"][i];

            // subtasks inherit everything else (infill mode, embedding mode, etc.)
            subtask_ids[i] = request_completion(subtask_data, multiprompt_task.infill_mode, multiprompt_task.embedding_mode, multitask_id);
        }

        // queue up the multitask so we can track its subtask progression
        add_multi_task(multitask_id, subtask_ids);
        return multitask_id;
    }

    void process_tasks()
    {
        std::lock_guard<std::mutex> lock(mutex_tasks);
        while (!queue_tasks.empty())
        {
            task_server task = queue_tasks.front();
            queue_tasks.erase(queue_tasks.begin());
            switch (task.type)
            {
                case COMPLETION_TASK: {
                    llama_client_slot *slot = get_slot(json_value(task.data, "slot_id", -1));
                    if (slot == nullptr)
                    {
                        LOG_TEE("slot unavailable\n");
                        // send error result
                        send_error(task, "slot unavailable");
                        return;
                    }

                    if (task.data.contains("system_prompt"))
                    {
                        process_system_prompt_data(task.data["system_prompt"]);
                    }

                    slot->reset();

                    slot->infill = task.infill_mode;
                    slot->embedding = task.embedding_mode;
                    slot->task_id = task.id;
                    slot->multitask_id = task.multitask_id;

                    if (!launch_slot_with_data(slot, task.data))
                    {
                        // send error result
                        send_error(task, "internal_error");
                        break;
                    }
                } break;
                case CANCEL_TASK: { // release slot linked with the task id
                    for (auto & slot : slots)
                    {
                        if (slot.task_id == task.target_id)
                        {
                            slot.release();
                            break;
                        }
                    }
                } break;
            }
        }

        // remove finished multitasks from the queue of multitasks, and add the corresponding result to the result queue
        auto queue_iterator = queue_multitasks.begin();
        while (queue_iterator != queue_multitasks.end())
        {
            if (queue_iterator->subtasks_remaining.empty())
            {
                // all subtasks done == multitask is done
                task_result aggregate_result;
                aggregate_result.id = queue_iterator->id;
                aggregate_result.stop = true;
                aggregate_result.error = false;

                // collect json results into one json result
                std::vector<json> result_jsons;
                for (auto& subres : queue_iterator->results)
                {
                    result_jsons.push_back(subres.result_json);
                    aggregate_result.error = aggregate_result.error && subres.error;
                }
                aggregate_result.result_json = json{ "results", result_jsons };

                std::lock_guard<std::mutex> lock(mutex_results);
                queue_results.push_back(aggregate_result);

                queue_iterator = queue_multitasks.erase(queue_iterator);
            }
            else
            {
                ++queue_iterator;
            }
        }
    }

    bool update_slots() {
        // attend tasks
        process_tasks();

        // update the system prompt wait until all slots are idle state
        if (system_need_update && all_slots_are_idle)
        {
            LOG_TEE("updating system prompt\n");
            update_system_prompt();
        }

        llama_batch_clear(batch);

        if (all_slots_are_idle)
        {
            if (system_prompt.empty() && clean_kv_cache)
            {
                LOG_TEE("all slots are idle and system prompt is empty, clear the KV cache\n");
                kv_cache_clear();
            }
            // avoid 100% usage of cpu all time
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        for (llama_client_slot &slot : slots)
        {
            if (slot.is_processing() && slot.cache_tokens.size() >= (size_t) slot.n_ctx)
            {
                // Shift context
                const int n_left    = slot.n_past - slot.params.n_keep - 1;
                const int n_discard = n_left / 2;

                LOG_TEE("slot %d: context shift - n_keep = %d, n_left = %d, n_discard = %d\n", slot.id, slot.params.n_keep, n_left, n_discard);
                llama_kv_cache_seq_rm   (ctx, slot.id, slot.params.n_keep + 1            , slot.params.n_keep + n_discard + 1);
                llama_kv_cache_seq_shift(ctx, slot.id, slot.params.n_keep + 1 + n_discard, slot.n_past, -n_discard);

                for (size_t i = slot.params.n_keep + 1 + n_discard; i < slot.cache_tokens.size(); i++)
                {
                    slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
                }

                slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);

                slot.n_past -= n_discard;

                slot.truncated = true;

                LOG_VERBOSE("context shift", {
                                                {"n_ctx",  n_ctx},
                                                {"n_keep", params.n_keep},
                                                {"n_left", n_left},
                                            });
            }
        }

        // decode any currently ongoing sequences
        for (auto & slot : slots)
        {
            // release the slot
            if (slot.command == RELEASE)
            {
                slot.state = IDLE;
                slot.command = NONE;
                slot.t_last_used = ggml_time_us();

                LOG_TEE("slot %d released (%d tokens in cache)\n", slot.id, (int) slot.cache_tokens.size());

                continue;
            }

            if (slot.state == IDLE)
            {
                continue;
            }

            slot.i_batch = batch.n_tokens;

            llama_batch_add(batch, slot.sampled, system_tokens.size() + slot.n_past, { slot.id }, true);

            slot.n_decoded += 1;
            slot.n_past += 1;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        // assign workload to the slots
        if (params.cont_batching || batch.n_tokens == 0)
        {
            for (auto & slot : slots)
            {
                const bool has_prompt = slot.prompt.is_array() || (slot.prompt.is_string() && !slot.prompt.get<std::string>().empty()) || !slot.images.empty();

                // empty prompt passed -> release the slot and send empty response
                if (slot.state == IDLE && slot.command == LOAD_PROMPT && !has_prompt)
                {
                    slot.release();
                    slot.print_timings();
                    send_final_response(slot);
                    continue;
                }

                // need process the prompt
                if (slot.state == IDLE && slot.command == LOAD_PROMPT)
                {
                    slot.state = PROCESSING;
                    slot.command = NONE;
                    std::vector<llama_token> prompt_tokens;
                    slot.t_start_process_prompt = ggml_time_us();
                    slot.t_start_genereration = 0;

                    if (slot.infill)
                    {
                        bool suff_rm_leading_spc = true;
                        if (params.input_suffix.find_first_of(' ') == 0 && params.input_suffix.size() > 1)
                        {
                            params.input_suffix.erase(0, 1);
                            suff_rm_leading_spc = false;
                        }
                        auto prefix_tokens = tokenize(slot.params.input_prefix, false);
                        auto suffix_tokens = tokenize(slot.params.input_suffix, false);

                        const int space_token = 29871; // TODO: this should not be hardcoded
                        if (suff_rm_leading_spc && !suffix_tokens.empty() && suffix_tokens[0] == space_token) {
                            suffix_tokens.erase(suffix_tokens.begin());
                        }

                        prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(model));
                        prefix_tokens.insert(prefix_tokens.begin(), llama_token_bos(model)); // always add BOS
                        prefix_tokens.insert(prefix_tokens.end(), llama_token_suffix(model));
                        prefix_tokens.insert(prefix_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
                        prefix_tokens.push_back(llama_token_middle(model));
                        prompt_tokens = prefix_tokens;
                    }
                    else
                    {
                        prompt_tokens = tokenize(slot.prompt, system_prompt.empty() && add_bos_token);  // add BOS if there isn't system prompt
                    }

                    slot.num_prompt_tokens = prompt_tokens.size();

                    if (slot.params.n_keep < 0)
                    {
                        slot.params.n_keep = slot.num_prompt_tokens;
                    }
                    slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                    // if input prompt is too big, truncate it
                    if (slot.num_prompt_tokens >= slot.n_ctx)
                    {
                        const int n_left = slot.n_ctx - slot.params.n_keep;
                        const int n_block_size = n_left / 2;
                        const int erased_blocks = (slot.num_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                        std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + slot.params.n_keep);
                        new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size, prompt_tokens.end());

                        LOG_VERBOSE("input truncated", {
                            {"n_ctx",  slot.n_ctx},
                            {"n_keep", slot.params.n_keep},
                            {"n_left", n_left},
                            {"new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend())},
                        });
                        slot.truncated = true;
                        prompt_tokens = new_tokens;

                        slot.num_prompt_tokens = prompt_tokens.size();
                        GGML_ASSERT(slot.num_prompt_tokens < slot.n_ctx);
                    }

                    if (!slot.params.cache_prompt)
                    {
                        llama_sampling_reset(slot.ctx_sampling);

                        slot.n_past = 0;
                        slot.num_prompt_tokens_processed = slot.num_prompt_tokens;
                    }
                    else
                    {
                        // push the prompt into the sampling context (do not apply grammar)
                        for (auto &token : prompt_tokens)
                        {
                            llama_sampling_accept(slot.ctx_sampling, ctx, token, false);
                        }

                        slot.n_past = common_part(slot.cache_tokens, prompt_tokens);
                        slot.num_prompt_tokens_processed = slot.num_prompt_tokens - slot.n_past;

                        LOG_TEE("slot %d : in cache: %i tokens | to process: %i tokens\n", slot.id, slot.n_past, slot.num_prompt_tokens_processed);
                    }

                    LOG_TEE("slot %d : kv cache rm - [%d, end)\n", slot.id, (int) system_tokens.size() + slot.n_past);

                    llama_kv_cache_seq_rm(ctx, slot.id, system_tokens.size() + slot.n_past, -1);

                    slot.cache_tokens = prompt_tokens;

                    if (slot.n_past == slot.num_prompt_tokens)
                    {
                        // we have to evaluate at least 1 token to generate logits.
                        LOG_TEE("slot %d : we have to evaluate at least 1 token to generate logits\n", slot.id);
                        slot.n_past--;
                    }

                    LOG_VERBOSE("prompt ingested", {
                                                    {"n_past", slot.n_past},
                                                    {"cached", tokens_to_str(ctx, slot.cache_tokens.cbegin(), slot.cache_tokens.cbegin() + slot.n_past)},
                                                    {"to_eval", tokens_to_str(ctx, slot.cache_tokens.cbegin() + slot.n_past, slot.cache_tokens.cend())},
                                                });

                    const bool has_images = process_images(slot);

                    // process the prefix of first image
                    std::vector<llama_token> prefix_tokens = has_images ? tokenize(slot.images[0].prefix_prompt, add_bos_token) : prompt_tokens;
                    for (; slot.n_past < (int) prefix_tokens.size(); ++slot.n_past)
                    {
                       llama_batch_add(batch, prefix_tokens[slot.n_past], system_tokens.size() + slot.n_past, { slot.id }, false);
                    }

                    if (has_images && !ingest_images(slot, n_batch))
                    {
                        LOG_TEE("failed processing images\n");
                        return false;
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0)
                    {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    slot.n_decoded = 0;
                    slot.i_batch   = batch.n_tokens - 1;
                }
            }
        }

        if (batch.n_tokens == 0)
        {
            all_slots_are_idle = true;
            return true;
        }

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch)
        {
            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));
            llama_batch batch_view =
            {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0)
            {
                if (n_batch == 1 || ret < 0)
                {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_TEE("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return false;
                }

                LOG_TEE("%s : failed to find free space in the KV cache, retrying with smaller n_batch = %d\n", __func__, n_batch / 2);

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;
                continue;
            }

            for (auto & slot : slots)
            {
                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens))
                {
                    continue;
                }

                // prompt evaluated for embedding
                if (slot.embedding)
                {
                    send_embedding(slot);
                    slot.release();
                    slot.i_batch = -1;
                    return true;
                }

                completion_token_output result;
                const llama_token id = llama_sampling_sample(slot.ctx_sampling, ctx, NULL, slot.i_batch - i);

                llama_sampling_accept(slot.ctx_sampling, ctx, id, true);

                if (slot.n_decoded == 1)
                {
                    slot.t_start_genereration = ggml_time_us();
                    slot.t_prompt_processing = (slot.t_start_genereration - slot.t_start_process_prompt) / 1e3;
                }

                llama_token_data_array cur_p = { slot.ctx_sampling->cur.data(), slot.ctx_sampling->cur.size(), false };
                result.tok = id;

                const int32_t n_probs = slot.sparams.n_probs;
                if (slot.sparams.temp <= 0 && n_probs > 0)
                {
                    // for llama_sample_token_greedy we need to sort candidates
                    llama_sample_softmax(ctx, &cur_p);
                }

                for (size_t i = 0; i < std::min(cur_p.size, (size_t)n_probs); ++i)
                {
                    result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
                }

                if (!process_token(result, slot))
                {
                    slot.release();
                    slot.print_timings();
                    send_final_response(slot);
                }

                slot.i_batch = -1;
            }
        }
        return true;
    }
};

class Alpaca::CImpl {
public:
	CImpl(std::filesystem::path& dataPath, const std::string& chatModel) {
		auto modelPath = std::filesystem::absolute(dataPath / (chatModel)).string();
		
		params.model = modelPath;
		params.model_alias = chatModel;
		llama_backend_init(params.numa);
		
		// load the model
		if (!llama.load_model(params)) {
			throw std::exception();
		}
		
		llama.initialize();
	}
	
	~CImpl() {
		// No need to manually join threads, unique_ptr will handle it
		threadPool.clear();
		llama_backend_free();
	}
	
	void ClearPool() {
		interactionPool.clear();
	}
	
	void ChatBetween(int interactionTurn, std::function<void(std::vector<std::pair<int, std::string>>)> callback, const std::string& playerName, const std::string& playerDescription, const std::string& npcName, const std::string& npcDescription, const std::string& playerPrompt = "", const std::string& npcPrompt = "", bool interactionEnd = false) {
		
		ChatBetweenInternal(interactionTurn, callback, playerName, playerDescription, npcName, npcDescription, playerPrompt, npcPrompt, interactionEnd);
	}
	
	bool Update() {
		bool result = llama.update_slots();

		if(!executionPool.empty()){
			slotMutex.lock();
			auto lastExecutor = executionPool.back();
			slotMutex.unlock();
			
			lastExecutor();
			
			slotMutex.lock();
			executionPool.pop_back();
			slotMutex.unlock();
		}
		

		
		for(auto thread_id : disposedThreadIdentifiers){
			erase_thread_from_pool(thread_id);
		}
		
		return result;
	}
	
private:
	void ChatBetweenInternal(int interactionTurn, std::function<void(std::vector<std::pair<int, std::string>>)> callback, const std::string& playerName, const std::string& playerDescription, const std::string& npcName, const std::string& npcDescription, const std::string& playerPrompt = "", const std::string& npcPrompt = "", bool interactionEnd = false) {
		
		nlohmann::json data;
		
		// Assign values field by field
		data["stream"] = true;
		data["n_predict"] = 400;
		data["temperature"] = 0.7;
		data["stop"] = { "</s>", npcName + ":", playerName + ":" };
		data["repeat_last_n"] = 256;
		data["repeat_penalty"] = 1.18;
		data["top_k"] = 40;
		data["top_p"] = 0.95;
		data["min_p"] = 0.05;
		data["tfs_z"] = 1;
		data["typical_p"] = 1;
		data["presence_penalty"] = 0;
		data["frequency_penalty"] = 0;
		data["mirostat"] = 0;
		data["mirostat_tau"] = 5;
		data["mirostat_eta"] = 0.1;
		data["grammar"] = "";
		data["n_probs"] = 0;
		data["image_data"] = json::array(); // Empty array for image_data
		data["cache_prompt"] = false;
		data["api_key"] = "";
		data["slot_id"] = interactionTurn;
		std::string totalPrompt = "Conversacion en Español entre " + playerName + " quien es " + playerDescription + " y " + npcName + " quien es " + npcDescription + ", la interacción es cordial y no lleva detalles de nacimiento o cosas que puedan cotejarse contra la realidad, como por ejemplo, datos escolares\n\n";
		
		if(interactionTurn == 0){
			if(npcPrompt.empty()){
				totalPrompt += playerName + ":" + playerPrompt;
			} else {
				totalPrompt += npcName + ":" + npcPrompt;
				
				totalPrompt += "\n\n";
				
				totalPrompt += playerName + ":" + playerPrompt;
			}
		} else {
			if(playerPrompt.empty()){
				totalPrompt += npcName + ":" + npcPrompt;
			} else {
				totalPrompt += playerName + ":" + playerPrompt;
				
				totalPrompt += "\n\n";
				
				totalPrompt += npcName + ":" + npcPrompt;
				
			}
		}
		
		data["prompt"] = totalPrompt;
		
		interactionPool.push_back({interactionTurn, ""});
		
		auto executor = std::function<void()>([this, playerName, playerDescription, npcName, npcDescription, playerPrompt, npcPrompt, interactionTurn, interactionEnd, data, callback](){
			
			const int task_id = llama.request_completion(data, false, false, -1);
					
			auto threadPtr = std::unique_ptr<std::thread>(new std::thread([this, playerName, playerDescription, npcName, npcDescription, playerPrompt, npcPrompt, interactionTurn, interactionEnd, task_id, callback]() {
				std::string output = "";
				
				while (true) {
					task_result result = llama.next_result(task_id);
					if (!result.error) {
						const std::string str =
						result.result_json["content"];
						
						outputMutex.lock();
						interactionPool.back().second += str;
						outputMutex.unlock();
						
						
						if (result.stop) {
							break;
						}
					} else {
						break;
					}
				}
				
				
				output = interactionPool.back().second;
				LOG_TEE("data stream %s", output.c_str());
				
				disposalMutex.lock();
				disposedThreadIdentifiers.push_back(task_id);
				disposalMutex.unlock();
				
				llama.request_cancel(task_id);
				
				if(!interactionEnd){
					if(interactionTurn == 0){
						ChatBetweenInternal(1, callback, playerName, playerDescription, npcName, npcDescription, output, "", true);
					} else {
						ChatBetweenInternal(0, callback, playerName, playerDescription, npcName, npcDescription, "", output, true);
						
					}
				} else {
					callback(interactionPool);
				}
			}));
			
			// Insert the unique_ptr of the thread into the map
			
			threadMutex.lock();
			threadPool[task_id] = std::move(threadPtr);
			threadMutex.unlock();
		});
		
		slotMutex.lock();
		executionPool.push_back(std::move(executor));
		slotMutex.unlock();
	}
	

	
	// Function to erase the thread from the threadPool
	void erase_thread_from_pool(int task_id) {
		auto it = threadPool.find(task_id);
		if (it != threadPool.end()) {
			it->second->join();
			
			disposalMutex.lock();
			threadPool.erase(it);
			disposalMutex.unlock();
		}
	}

	
	int lastConversationId = 0;

	// own arguments required by this example
	gpt_params params;
	
	// struct that contains llama context and inference
	llama_server_context llama;
	
	std::map<int, std::unique_ptr<std::thread>> threadPool;
	std::vector<std::pair<int, std::string>> interactionPool;
	std::vector<int> disposedThreadIdentifiers;
	
	std::mutex disposalMutex;
	std::mutex outputMutex;
	std::recursive_mutex slotMutex;
	std::mutex threadMutex;
	
	std::vector<std::function<void()>> executionPool;
};

Alpaca::Alpaca(std::filesystem::path& dataPath, const std::string& chatModel) : impl(std::unique_ptr<CImpl>(new CImpl(dataPath, chatModel))){
}

Alpaca::~Alpaca(){
	
}

void Alpaca::Chat(int interactionTurn, std::function<void(std::vector<std::pair<int, std::string>>)> callback, const std::string& playerName, const std::string& playerDescription, const std::string& npcName, const std::string& npcDescription, const std::string& playerPrompt, const std::string& npcPrompt, bool interactionEnd)

{
	impl->ClearPool();
	
	impl->ChatBetween(interactionTurn, callback, playerName, playerDescription, npcName, npcDescription, playerPrompt, npcPrompt, interactionEnd);
}

bool Alpaca::Update() {
	return impl->Update();
}
