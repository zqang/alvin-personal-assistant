#include "CMiniCPM.h"
#include <llama/llama.h>
#include <llama/mtmd.h>
#include <llama/mtmd-helper.h>
#include <llama/speculative.h>
#include <llama/llama-ext.h>
#include <llama/gguf.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct minicpm_control {
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    std::condition_variable idle;
    int active = 0;
};

struct Operation {
    minicpm_control * control;
    explicit Operation(minicpm_control * c) : control(c) {
        std::lock_guard<std::mutex> lock(control->mutex);
        ++control->active;
    }
    ~Operation() {
        std::lock_guard<std::mutex> lock(control->mutex);
        --control->active;
        control->idle.notify_all();
    }
};

struct Cancelled : std::exception {};
static void check_cancelled(minicpm_control * c) {
    if (c->cancelled.load(std::memory_order_relaxed)) throw Cancelled();
}
static bool abort_eval(void * data) {
    return static_cast<minicpm_control *>(data)->cancelled.load(std::memory_order_relaxed);
}
static bool load_progress(float, void * data) { return !abort_eval(data); }

struct EvalControl {
    minicpm_control * cancellation;
    int interval = 1;
};

// The pinned fork only exposes the abort callback through CPU's backend registry.
// Check cancellation at every scheduler node, synchronizing periodically rather
// than every layer. A cancellation requests a checkpoint at the next node.
static bool eval_checkpoint(ggml_tensor * tensor, bool ask, void * data) {
    const auto * eval = static_cast<EvalControl *>(data);
    if (!ask) return !abort_eval(eval->cancellation);
    if (abort_eval(eval->cancellation)) return true;
    const char * name = ggml_get_name(tensor);
    const char * suffix = nullptr;
    if (std::strncmp(name, "l_out-", 6) == 0) suffix = name + 6;
    else if (std::strncmp(name, "layer_out", 9) == 0) suffix = name + 9;
    if (!suffix) return false;
    while (*suffix == '-' || *suffix == '_') ++suffix;
    char * end = nullptr;
    const long layer = std::strtol(suffix, &end, 10);
    return end == suffix || (layer + 1) % eval->interval == 0;
}

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Vision = std::unique_ptr<mtmd_context, decltype(&mtmd_free)>;
struct minicpm_session {
    minicpm_control * control;
    Model model{nullptr, llama_model_free};
    Context context{nullptr, llama_free};
    Vision vision{nullptr, mtmd_free};
    // Destruction order matters: draft context borrows the target's embeddings.
    Model draft_model{nullptr, llama_model_free};
    Context draft_context{nullptr, llama_free};
    common_speculative_ptr speculative;
    std::string experts_path;
    bool les_enabled = false;
    minicpm_runtime_config config = minicpm_default_runtime_config();
    EvalControl eval;
    explicit minicpm_session(minicpm_control * c) : control(c), eval{c, 1} {}
    ~minicpm_session() {
        if (context) llama_synchronize(context.get());
        if (draft_context) llama_synchronize(draft_context.get());
    }
};

static char * copy_string(const std::string & value) {
    char * copy = static_cast<char *>(std::malloc(value.size() + 1));
    if (!copy) throw std::bad_alloc();
    std::memcpy(copy, value.c_str(), value.size() + 1);
    return copy;
}
static void set_error(char ** error, const char * value) noexcept {
    if (!error) return;
    *error = ::strdup(value);
}

minicpm_control * minicpm_control_create() {
    try { return new minicpm_control; } catch (...) { return nullptr; }
}
void minicpm_control_free(minicpm_control * c) { delete c; }
void minicpm_cancel(minicpm_control * c) { if (c) c->cancelled.store(true, std::memory_order_relaxed); }
bool minicpm_is_cancelled(const minicpm_control * c) { return c && c->cancelled.load(std::memory_order_relaxed); }
bool minicpm_wait_until_idle(minicpm_control * c, double timeout) {
    if (!c) return true;
    std::unique_lock<std::mutex> lock(c->mutex);
    return c->idle.wait_for(lock, std::chrono::duration<double>(std::max(0.0, timeout)),
                           [c] { return c->active == 0; });
}
void minicpm_string_free(char * value) { std::free(value); }

minicpm_session * minicpm_load(const char * model_path, const char * projector_path,
                              minicpm_control * control, char ** error) {
    return minicpm_load_configured(model_path, projector_path, nullptr, nullptr, control, error);
}

minicpm_session * minicpm_load_configured(const char * model_path, const char * projector_path,
                                         const char * experts_manifest_path, const char * draft_path,
                                         minicpm_control * control, char ** error) {
    return minicpm_load_with_runtime_config(model_path, projector_path, experts_manifest_path, draft_path,
                                            minicpm_default_runtime_config(), control, error);
}

minicpm_runtime_config minicpm_default_runtime_config(void) { return {1, 3, 32, false, 6, 4}; }

minicpm_session * minicpm_load_with_runtime_config(const char * model_path, const char * projector_path,
                                         const char * experts_manifest_path, const char * draft_path,
                                         minicpm_runtime_config config,
                                         minicpm_control * control, char ** error) {
    if (error) *error = nullptr;
    if (!control || !model_path) {
        set_error(error, "Model and cancellation control are required."); return nullptr;
    }
    Operation operation(control);
    try {
        check_cancelled(control);
        static std::once_flag backend_once;
        std::call_once(backend_once, [] { llama_backend_init(); });
        auto session = std::make_unique<minicpm_session>(control);
        if (config.checkpoint_layer_interval < 1 || config.checkpoint_layer_interval > 8 ||
            config.draft_max < 1 || config.draft_max > 3 ||
            config.expert_cache_size < 32 || config.expert_cache_size > 80 ||
            config.cpu_threads < 1 || config.cpu_threads > 8 ||
            config.io_threads < 1 || config.io_threads > 16)
            throw std::runtime_error("Runtime profile is outside the validated tuning bounds.");
        session->config = config;
        session->eval.interval = config.checkpoint_layer_interval;
        const bool les = experts_manifest_path && *experts_manifest_path;
        const bool dspark = draft_path && *draft_path;
        if (dspark && !les) throw std::runtime_error("DSpark requires the matching 35B-A3B LES model and expert manifest.");
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;
        model_params.progress_callback = load_progress;
        model_params.progress_callback_user_data = control;
        if (les) {
            // The fork memoizes this pool size on its first read. Refuse a
            // misleading in-process profile switch rather than reporting an
            // unimplemented setting. All profiles must use one IO width per app launch.
            static std::mutex io_configuration_mutex;
            static int process_io_threads = 0;
            {
                std::lock_guard<std::mutex> lock(io_configuration_mutex);
                if (process_io_threads != 0 && process_io_threads != config.io_threads)
                    throw std::runtime_error("Changing LES I/O threads requires a fresh app process. Relaunch before selecting this profile.");
                setenv("LLAMA_MOE_IO_THREADS", std::to_string(config.io_threads).c_str(), 1);
                process_io_threads = config.io_threads;
            }
            session->experts_path = experts_manifest_path;
            model_params.moe_experts_path = session->experts_path.c_str();
            model_params.moe_slot_bank = 256; // Shared prefill bank; trained top-8 routing remains unchanged.
            model_params.use_mmap = true;
            // The per-layer decode cache must cover the entire verification block.
            // These are process-wide runtime settings; the app serializes all GPU engines.
            setenv("LLAMA_MOE_DECODE_WINDOW", std::to_string(config.expert_cache_size).c_str(), 1);
            setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1", 1);
            setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "4", 1);
            setenv("LLAMA_MOE_WINDOW_MAX_ROWS", "4", 1);
        }
        session->model.reset(llama_model_load_from_file(model_path, model_params));
        check_cancelled(control);
        if (!session->model) throw std::runtime_error("Could not load MiniCPM-V GGUF model and requested experts.");
        if (les) {
            char architecture[64] = {};
            llama_model_meta_val_str(session->model.get(), "general.architecture", architecture, sizeof(architecture));
            if (std::strcmp(architecture, "qwen35moe") != 0)
                throw std::runtime_error("LES configuration requires the 35B-A3B qwen35moe split model.");
            session->les_enabled = true;
        }

        const int threads = static_cast<int>(std::max(1u, std::min(static_cast<unsigned>(config.cpu_threads), std::thread::hardware_concurrency())));
        session->config.cpu_threads = threads;
        auto context_params = llama_context_default_params();
        context_params.n_ctx = 4096;
        context_params.n_batch = 512;
        context_params.n_ubatch = 128;
        context_params.n_rs_seq = dspark ? config.draft_max : 0;
        context_params.n_threads = context_params.n_threads_batch = threads;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_params.abort_callback = abort_eval;
        context_params.abort_callback_data = control;
        context_params.cb_eval = eval_checkpoint;
        context_params.cb_eval_user_data = &session->eval;
        session->context.reset(llama_init_from_model(session->model.get(), context_params));
        check_cancelled(control);
        if (!session->context) throw std::runtime_error("Could not allocate MiniCPM context.");

        if (dspark) {
            // A plain DFlash checkpoint has the same architecture name but no
            // DSpark Markov head. Never label that alternative as DSpark.
            std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
                gguf_init_from_file(draft_path, {true, nullptr}), gguf_free);
            if (!metadata || gguf_find_tensor(metadata.get(), "markov_w1.weight") < 0 ||
                gguf_find_tensor(metadata.get(), "markov_w2.weight") < 0 ||
                gguf_find_tensor(metadata.get(), "conf_proj.weight") < 0)
                throw std::runtime_error("Draft GGUF is missing the required DSpark Markov/confidence head.");
            // This probe clears KV; it MUST run before the first prompt.
            const auto rollback = common_context_can_seq_rm(session->context.get());
            if (rollback != COMMON_CONTEXT_SEQ_RM_TYPE_RS && rollback != COMMON_CONTEXT_SEQ_RM_TYPE_PART)
                throw std::runtime_error("DSpark target does not support bounded recurrent-state rollback.");
            auto draft_params = llama_model_default_params();
            draft_params.n_gpu_layers = 99;
            draft_params.progress_callback = load_progress;
            draft_params.progress_callback_user_data = control;
            session->draft_model.reset(llama_model_load_from_file(draft_path, draft_params));
            check_cancelled(control);
            if (!session->draft_model) throw std::runtime_error("Could not load the required DSpark draft GGUF.");
            char draft_arch[64] = {};
            llama_model_meta_val_str(session->draft_model.get(), "general.architecture", draft_arch, sizeof(draft_arch));
            if (std::strcmp(draft_arch, "dflash") != 0 || llama_model_target_layer_ids_n(session->draft_model.get()) == 0)
                throw std::runtime_error("The draft is not a DSpark/DFlash model with target hidden-state extraction metadata.");
            const auto * layers = llama_model_target_layer_ids(session->draft_model.get());
            for (uint32_t i = 0; i < llama_model_target_layer_ids_n(session->draft_model.get()); ++i)
                if (layers[i] < 0 || layers[i] >= llama_model_n_layer(session->model.get()))
                    throw std::runtime_error("DSpark extraction layer is outside the target model.");
            auto draft_context_params = context_params;
            draft_context_params.n_rs_seq = 0; // DFlash is not recurrent.
            draft_context_params.ctx_other = session->context.get(); // Shared target embeddings and output head.
            session->draft_context.reset(llama_init_from_model(session->draft_model.get(), draft_context_params));
            check_cancelled(control);
            if (!session->draft_context) throw std::runtime_error("Could not create the DSpark draft context.");
            common_params_speculative spec_params;
            spec_params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
            spec_params.draft.mparams.path = draft_path;
            spec_params.draft.ctx_tgt = session->context.get();
            spec_params.draft.ctx_dft = session->draft_context.get();
            spec_params.draft.n_max = config.draft_max;
            spec_params.draft.n_min = 0;
            spec_params.draft.p_min = 0;
            spec_params.draft.backend_sampling = false;
            session->speculative.reset(common_speculative_init(spec_params, 1));
            if (!session->speculative) throw std::runtime_error("DSpark initialization failed; speculation was not enabled.");
        }

        if (projector_path && *projector_path) {
            auto vision_params = mtmd_context_params_default();
            vision_params.use_gpu = true;
            vision_params.n_threads = threads;
            vision_params.warmup = false;
            vision_params.print_timings = false;
            vision_params.image_max_slice_nums = 1;
            vision_params.cb_eval = eval_checkpoint;
            vision_params.cb_eval_user_data = &session->eval;
            session->vision.reset(mtmd_init_from_file(projector_path, session->model.get(), vision_params));
            check_cancelled(control);
            if (!session->vision || !mtmd_support_vision(session->vision.get())) {
                throw std::runtime_error("Could not load the matching MiniCPM vision projector.");
            }
        }
        return session.release();
    } catch (const Cancelled &) { set_error(error, "Cancelled."); }
      catch (const std::exception & e) { set_error(error, e.what()); }
      catch (...) { set_error(error, "Unknown MiniCPM model loading failure."); }
    return nullptr;
}

void minicpm_free(minicpm_session * session) {
    if (!session) return;
    Operation operation(session->control);
    delete session;
}

// A token can contain only part of a UTF-8 scalar. Do not call that first text.
static bool has_complete_visible_text(const std::string & text) {
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        size_t length = c < 0x80 ? 1 : c >= 0xC2 && c < 0xE0 ? 2 : c < 0xF0 && c >= 0xE0 ? 3 : c <= 0xF4 && c >= 0xF0 ? 4 : 0;
        if (!length) { ++i; continue; }
        if (i + length > text.size()) return false;
        bool valid = true;
        for (size_t j = 1; j < length; ++j) valid &= (static_cast<unsigned char>(text[i + j]) & 0xC0) == 0x80;
        if (valid && (length > 1 || (c > 0x20 && c != 0x7F))) return true;
        i += valid ? length : 1;
    }
    return false;
}

struct Batch {
    llama_batch value;
    explicit Batch(int capacity) : value(llama_batch_init(capacity, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
    void add(llama_token token, llama_pos position, bool logits) {
        const int i = value.n_tokens++;
        value.token[i] = token;
        value.pos[i] = position;
        value.n_seq_id[i] = 1;
        value.seq_id[i][0] = 0;
        value.logits[i] = logits;
    }
};

static void evaluate(minicpm_session * s, const llama_batch & batch, bool feed_draft) {
    check_cancelled(s->control);
    const int rc = llama_decode(s->context.get(), batch);
    llama_synchronize(s->context.get());
    check_cancelled(s->control);
    if (rc != 0) throw std::runtime_error("MiniCPM target evaluation failed.");
    if (feed_draft && s->speculative) {
        // Layer inputs belong to the most recent physical microbatch. Call this
        // before another target decode can overwrite them (including prefill).
        const bool success = common_speculative_process(s->speculative.get(), batch);
        llama_synchronize(s->draft_context.get());
        check_cancelled(s->control);
        if (!success) throw std::runtime_error("DSpark hidden-state/KV injection failed.");
    }
}

static void trim_memory(llama_context * context, llama_pos position) {
    if (!llama_memory_seq_rm(llama_get_memory(context), 0, position, -1))
        throw std::runtime_error("DSpark KV/recurrent-state rollback failed.");
}

int32_t minicpm_run(minicpm_session * s, const char * prompt, const char * image_path,
                    int32_t max_tokens, minicpm_result * result, char ** error) {
    if (error) *error = nullptr;
    if (!s || !prompt || !result) { set_error(error, "A loaded session and prompt are required."); return 1; }
    Operation operation(s->control);
    try {
        check_cancelled(s->control);
        if (max_tokens < 1 || max_tokens > 2048) throw std::runtime_error("Token limit must be between 1 and 2048.");
        if (!*prompt || std::strlen(prompt) > 32768) throw std::runtime_error("Prompt must contain 1–32768 UTF-8 bytes.");
        if (std::strstr(prompt, mtmd_default_marker())) throw std::runtime_error("Prompt cannot contain the reserved image marker.");

        *result = {};
        result->first_text_seconds = -1;
        result->les_enabled = s->les_enabled;
        result->dspark_enabled = bool(s->speculative);
        result->runtime_config = s->config;
        llama_memory_clear(llama_get_memory(s->context.get()), true);
        if (s->draft_context) llama_memory_clear(llama_get_memory(s->draft_context.get()), true);
        const auto start = Clock::now();
        std::string formatted = "<|im_start|>user\n";
        const bool has_image = image_path && *image_path;
        if (has_image && !s->vision) throw std::runtime_error("This text-only session has no vision projector.");
        if (has_image && s->speculative)
            throw std::runtime_error("DSpark benchmark currently supports text prompts only; image feature injection is not implemented.");
        if (has_image) formatted += std::string(mtmd_default_marker()) + "\n";
        formatted += std::string(prompt) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        llama_pos next_position = 0;
        llama_tokens prompt_history;
        if (!has_image) {
            const auto * vocab = llama_model_get_vocab(s->model.get());
            int n = llama_tokenize(vocab, formatted.data(), static_cast<int32_t>(formatted.size()), nullptr, 0, true, true);
            if (n >= 0) throw std::runtime_error("MiniCPM text tokenization failed.");
            prompt_history.resize(static_cast<size_t>(-n));
            n = llama_tokenize(vocab, formatted.data(), static_cast<int32_t>(formatted.size()), prompt_history.data(), -n, true, true);
            if (n <= 0 || static_cast<uint64_t>(n) + max_tokens + s->config.draft_max > llama_n_ctx(s->context.get()))
                throw std::runtime_error("Prompt and output exceed the 4096-token context.");
            prompt_history.resize(static_cast<size_t>(n));
            result->prompt_tokens = n;
            const int ubatch = static_cast<int>(llama_n_ubatch(s->context.get()));
            Batch batch(ubatch);
            for (int offset = 0; offset < n; offset += ubatch) {
                batch.value.n_tokens = 0;
                for (int i = offset; i < std::min(offset + ubatch, n); ++i)
                    batch.add(prompt_history[i], next_position++, i + 1 == n);
                evaluate(s, batch.value, true);
            }
            if (s->speculative) common_speculative_begin(s->speculative.get(), 0, prompt_history);
        } else {
            std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(nullptr, mtmd_bitmap_free);
            if (has_image) {
                bitmap.reset(mtmd_helper_bitmap_init_from_file(s->vision.get(), image_path));
                if (!bitmap || mtmd_bitmap_is_audio(bitmap.get())) throw std::runtime_error("Could not decode the benchmark image. Use PNG or JPEG.");
            }
            check_cancelled(s->control);
            std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
            if (!chunks) throw std::bad_alloc();
            mtmd_input_text input{formatted.c_str(), true, true};
            const mtmd_bitmap * bitmaps[] = {bitmap.get()};
            if (mtmd_tokenize(s->vision.get(), chunks.get(), &input, has_image ? bitmaps : nullptr, has_image ? 1 : 0) != 0)
                throw std::runtime_error("MiniCPM image/prompt tokenization failed.");
            const auto prompt_tokens = mtmd_helper_get_n_tokens(chunks.get());
            const auto positions = mtmd_helper_get_n_pos(chunks.get());
            if (positions < 1 || static_cast<uint64_t>(positions) + max_tokens > llama_n_ctx(s->context.get()))
                throw std::runtime_error("Prompt and output exceed the 4096-token context. Shorten the prompt or output limit.");
            result->prompt_tokens = static_cast<int32_t>(prompt_tokens);
            const size_t count = mtmd_input_chunks_size(chunks.get());
            for (size_t i = 0; i < count; ++i) {
                check_cancelled(s->control);
                const auto * chunk = mtmd_input_chunks_get(chunks.get(), i);
                int status;
                if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    status = mtmd_encode_chunk(s->vision.get(), chunk);
                    // A scheduler cancellation can return without all image tensors
                    // being computed. Never send those partial embeddings to the LLM.
                    check_cancelled(s->control);
                    if (status == 0) {
                        status = mtmd_helper_decode_image_chunk(s->vision.get(), s->context.get(), chunk,
                            mtmd_get_output_embd(s->vision.get()), next_position, 0, 512, &next_position);
                    }
                } else {
                    status = mtmd_helper_eval_chunk_single(s->vision.get(), s->context.get(), chunk,
                        next_position, 0, 512, i + 1 == count, &next_position);
                }
                llama_synchronize(s->context.get());
                check_cancelled(s->control);
                if (status != 0) throw std::runtime_error("MiniCPM prompt evaluation failed.");
            }
        }
        result->prefill_seconds = seconds(start);
        auto sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(llama_sampler_init_greedy(), llama_sampler_free);
        if (!sampler) throw std::bad_alloc();
        const llama_vocab * vocab = llama_model_get_vocab(s->model.get());
        auto sample = [&](int32_t row) {
            if (!s->config.direct_greedy) return llama_sampler_sample(sampler.get(), s->context.get(), row);
            const float * logits = llama_get_logits_ith(s->context.get(), row);
            if (!logits) throw std::runtime_error("Target logits unavailable.");
            // Identical ascending token order and strict > comparison to the
            // upstream greedy sampler, without constructing a full candidate array.
            llama_token best = 0;
            const int vocabulary_size = llama_vocab_n_tokens(vocab);
            for (llama_token id = 1; id < vocabulary_size; ++id)
                if (logits[id] > logits[best]) best = id;
            return best;
        };
        std::string output;
        std::vector<char> piece(256);
        result->finish_reason = 1;
        auto emit = [&](llama_token token) {
            if (llama_vocab_is_eog(vocab, token)) { result->finish_reason = 0; return false; }
            int bytes = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
            if (bytes < 0) {
                piece.resize(static_cast<size_t>(-bytes));
                bytes = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
            }
            if (bytes < 0) throw std::runtime_error("MiniCPM token decoding failed.");
            output.append(piece.data(), static_cast<size_t>(bytes));
            ++result->generated_tokens;
            if (result->first_text_seconds < 0 && has_complete_visible_text(output)) result->first_text_seconds = seconds(start);
            return result->generated_tokens < max_tokens;
        };
        if (s->speculative) {
            llama_token anchor = sample(-1);
            bool more = emit(anchor);
            Batch verify(s->config.draft_max + 1);
            while (more) {
                check_cancelled(s->control);
                const auto decode_start = Clock::now();
                llama_tokens draft;
                const int remaining = max_tokens - result->generated_tokens;
                if (remaining > 1) {
                    auto & dp = common_speculative_get_draft_params(s->speculative.get(), 0);
                    dp.drafting = true;
                    dp.n_max = std::min(s->config.draft_max, remaining - 1);
                    dp.n_past = next_position;
                    dp.id_last = anchor;
                    dp.prompt = &prompt_history;
                    dp.result = &draft;
                    const auto draft_start = Clock::now();
                    common_speculative_draft(s->speculative.get());
                    llama_synchronize(s->draft_context.get());
                    check_cancelled(s->control);
                    result->draft_seconds += seconds(draft_start);
                    // The fork uses its configured width rather than dp.n_max.
                    // Bound both verification storage and remaining output budget.
                    const size_t draft_limit = static_cast<size_t>(std::min(s->config.draft_max, remaining - 1));
                    if (draft.size() > draft_limit) draft.resize(draft_limit);
                    if (draft.empty()) throw std::runtime_error("DSpark produced no draft tokens; refusing silent target-only fallback.");
                    // Remove noise tokens before injecting real target features at
                    // those same M-RoPE positions (official app/server sequence).
                    trim_memory(s->draft_context.get(), next_position);
                }
                verify.value.n_tokens = 0;
                verify.add(anchor, next_position, true);
                for (size_t i = 0; i < draft.size(); ++i) verify.add(draft[i], next_position + 1 + i, true);
                evaluate(s, verify.value, true);
                llama_tokens ids;
                int accepted = 0;
                for (size_t i = 0; i <= draft.size(); ++i) {
                    const llama_token id = sample(static_cast<int32_t>(i));
                    ids.push_back(id);
                    // Accept only a contiguous exact argmax prefix. The last id is
                    // always a target correction/bonus, pending next round's decode.
                    if (llama_vocab_is_eog(vocab, id) || i == draft.size() || id != draft[i]) break;
                    ++accepted;
                }
                next_position += 1 + accepted;
                trim_memory(s->context.get(), next_position);
                trim_memory(s->draft_context.get(), next_position);
                if (!draft.empty()) common_speculative_accept(s->speculative.get(), 0, static_cast<uint16_t>(accepted));
                prompt_history.push_back(anchor);
                prompt_history.insert(prompt_history.end(), draft.begin(), draft.begin() + accepted);
                if (!draft.empty()) ++result->speculation_rounds;
                result->draft_tokens += static_cast<int32_t>(draft.size());
                result->accepted_draft_tokens += accepted;
                result->decoded_tokens += 1 + accepted;
                result->decode_seconds += seconds(decode_start);
                for (llama_token id : ids) { if (!(more = emit(id))) break; }
                anchor = ids.back();
            }
        } else for (int i = 0; i < max_tokens; ++i) {
            check_cancelled(s->control);
            const llama_token token = sample(-1);
            if (!emit(token)) break;
            // Last sampled token does not require a forward pass. Count evaluated
            // tokens separately so short outputs cannot inflate the decode rate.
            if (i + 1 == max_tokens) break;
            const auto decode_start = Clock::now();
            llama_token input_token = token;
            llama_batch batch = llama_batch_get_one(&input_token, 1);
            const int status = llama_decode(s->context.get(), batch);
            llama_synchronize(s->context.get());
            check_cancelled(s->control);
            if (status != 0) throw std::runtime_error("MiniCPM token evaluation failed.");
            result->decode_seconds += seconds(decode_start);
            ++result->decoded_tokens;
        }
        result->total_seconds = seconds(start);
        result->text = copy_string(output);
        return 0;
    } catch (const Cancelled &) {
        llama_synchronize(s->context.get());
        if (s->draft_context) llama_synchronize(s->draft_context.get());
        set_error(error, "Cancelled."); return 2;
    } catch (const std::exception & e) {
        llama_synchronize(s->context.get());
        if (s->draft_context) llama_synchronize(s->draft_context.get());
        set_error(error, e.what()); return 1;
    } catch (...) {
        llama_synchronize(s->context.get());
        if (s->draft_context) llama_synchronize(s->draft_context.get());
        set_error(error, "Unknown MiniCPM inference failure."); return 1;
    }
}
