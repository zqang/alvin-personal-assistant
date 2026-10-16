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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
    // Whether the run in progress was cancelled with minicpm_cancel_run. Every evaluation callback reads it.
    std::atomic<bool> run_cancelled{false};
    std::atomic<uint64_t> last_run{0};
    std::mutex mutex;
    std::condition_variable idle;
    int active = 0;
    uint64_t current_run = 0;            // Guarded by mutex.
    std::deque<uint64_t> cancelled_runs; // Guarded by mutex: the most recent cancellations, for runs not started yet.
};

// Native work in progress, for minicpm_wait_until_idle. A call that belongs to a run also makes
// that run's cancellation visible to the evaluation callbacks while it lasts.
struct Operation {
    minicpm_control * control;
    const uint64_t run;
    explicit Operation(minicpm_control * c, uint64_t r = 0) : control(c), run(r) {
        std::lock_guard<std::mutex> lock(control->mutex);
        ++control->active;
        if (run) {
            control->current_run = run;
            const auto & runs = control->cancelled_runs;
            control->run_cancelled.store(std::find(runs.begin(), runs.end(), run) != runs.end());
        }
    }
    ~Operation() {
        std::lock_guard<std::mutex> lock(control->mutex);
        if (run) {
            control->current_run = 0;
            control->run_cancelled.store(false);
        }
        --control->active;
        control->idle.notify_all();
    }
};

struct Cancelled : std::exception {};
static bool is_cancelled(const minicpm_control * c) {
    return c->cancelled.load(std::memory_order_relaxed) || c->run_cancelled.load(std::memory_order_relaxed);
}
static void check_cancelled(minicpm_control * c) {
    if (is_cancelled(c)) throw Cancelled();
}
static bool abort_eval(void * data) {
    return is_cancelled(static_cast<minicpm_control *>(data));
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

// A hybrid model's recurrent state at `position` (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY). The
// attention cache needs no copy: its entries before `position` never change.
struct Checkpoint {
    int32_t position = -1;
    std::vector<uint8_t> state;
};

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
    bool recurrent = false;
    // The vocabulary's USER_DEFINED token texts (<think>, <tool_call>…): llama_tokenize matches
    // them even with parse_special false, so minicpm_tokenize splits them out of plain text.
    std::vector<std::string> user_defined;
    minicpm_runtime_config config = minicpm_default_runtime_config();
    EvalControl eval;

    // Reuse state of the session API (see CMiniCPM.h).
    std::vector<llama_token> ledger;          // Fed tokens, at positions 0..<ledger.size().
    llama_token pending = LLAMA_TOKEN_NULL;   // Emitted, not fed yet.
    bool logits_ready = false;                // The last decode left the logits of ledger.back().
    bool dirty = false;                       // An interrupted decode may have left state the ledger doesn't describe.
    bool foreign = false;                     // minicpm_run or minicpm_describe_image used the cache.
    Checkpoint checkpoints[3];
    uint64_t checkpoint_bytes = 0;

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

uint64_t minicpm_begin_run(minicpm_control * c) { return c ? c->last_run.fetch_add(1) + 1 : 0; }

void minicpm_cancel_run(minicpm_control * c, uint64_t run) {
    if (!c || !run) return;
    std::lock_guard<std::mutex> lock(c->mutex);
    // ponytail: remembers the last 64 cancellations; a run cancelled before it starts must start
    // before 64 more runs are cancelled. Runs are serialized per session, so a few are ever queued.
    c->cancelled_runs.push_back(run);
    if (c->cancelled_runs.size() > 64) c->cancelled_runs.pop_front();
    if (c->current_run == run) c->run_cancelled.store(true);
}

bool minicpm_run_is_cancelled(minicpm_control * c, uint64_t run) {
    if (!c) return true;
    if (c->cancelled.load(std::memory_order_relaxed)) return true;
    std::lock_guard<std::mutex> lock(c->mutex);
    return std::find(c->cancelled_runs.begin(), c->cancelled_runs.end(), run) != c->cancelled_runs.end();
}

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
        session->recurrent = llama_model_is_recurrent(session->model.get()) || llama_model_is_hybrid(session->model.get());
        const auto * vocab = llama_model_get_vocab(session->model.get());
        for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token)
            if ((llama_vocab_get_attr(vocab, token) & LLAMA_TOKEN_ATTR_USER_DEFINED) && *llama_vocab_get_text(vocab, token))
                session->user_defined.emplace_back(llama_vocab_get_text(vocab, token));

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

// The length of the longest prefix of `text` that doesn't end inside a multi-byte UTF-8 sequence.
static size_t complete_utf8_prefix(const std::string & text) {
    const size_t n = text.size();
    for (size_t back = 1; back <= std::min<size_t>(n, 4); ++back) {
        const unsigned char c = static_cast<unsigned char>(text[n - back]);
        if ((c & 0xC0) == 0x80) continue;
        const size_t length = c < 0xC0 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        return back >= length ? n : n - back;
    }
    return n;
}

static void append_piece(const llama_vocab * vocab, llama_token token, std::vector<char> & piece, std::string & output) {
    int bytes = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
    if (bytes < 0) {
        piece.resize(static_cast<size_t>(-bytes));
        bytes = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
    }
    if (bytes < 0) throw std::runtime_error("MiniCPM token decoding failed.");
    output.append(piece.data(), static_cast<size_t>(bytes));
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

// Evaluates `batch`. It throws unless the whole batch is in the cache (and the draft's), with no
// cancellation pending; then the session is dirty, since a stopped graph or a failed rollback can
// leave part of the batch in a hybrid model's state.
static void evaluate(minicpm_session * s, const llama_batch & batch, bool feed_draft) {
    check_cancelled(s->control);
    int rc;
    try {
        rc = llama_decode(s->context.get(), batch);
    } catch (...) {
        // A graph a cancellation stopped part-way can throw from the runtime (LES reads the
        // half-computed expert ids), leaving the batch's cells in the cache.
        s->dirty = true;
        llama_synchronize(s->context.get());
        check_cancelled(s->control);
        throw;
    }
    llama_synchronize(s->context.get());
    // A cancellation can stop the graph early and still return 0.
    if (rc != 0 || is_cancelled(s->control)) {
        s->dirty = true;
        check_cancelled(s->control);
        throw std::runtime_error("MiniCPM target evaluation failed.");
    }
    if (feed_draft && s->speculative) {
        // Layer inputs belong to the most recent physical microbatch. Call this
        // before another target decode can overwrite them (including prefill).
        const bool success = common_speculative_process(s->speculative.get(), batch);
        llama_synchronize(s->draft_context.get());
        if (!success || is_cancelled(s->control)) {
            s->dirty = true;
            check_cancelled(s->control);
            throw std::runtime_error("DSpark hidden-state/KV injection failed.");
        }
    }
}

static void trim_memory(minicpm_session * s, llama_context * context, llama_pos position) {
    if (!llama_memory_seq_rm(llama_get_memory(context), 0, position, -1)) {
        s->dirty = true;
        throw std::runtime_error("DSpark KV/recurrent-state rollback failed.");
    }
}

// Greedy decoding from the logits of the last evaluated token, the next one at `position`. Feeds
// every emitted token except the last (and never an end of generation), appending what it feeds to
// `history`. `emit(token)` delivers a token's text and returns false to stop (a cancellation); every
// fed token is delivered, so a stop in a DSpark round takes effect after the round's accepted drafts.
// On return or throw, `unfed` is the last emitted token if it wasn't fed. `strict_draft` refuses a
// DSpark round without draft tokens (the benchmark's accounting); otherwise that round verifies the
// anchor alone, a plain target step.
template <typename Emit>
static void decode(minicpm_session * s, llama_pos position, llama_tokens & history, int32_t max_tokens,
                   minicpm_result * result, Emit & emit, llama_token & unfed, bool strict_draft) {
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
    result->finish_reason = 1;
    bool stopped = false;
    // Delivers `token`; false once it ends the generation or spends the budget. A stop throws at
    // the first unfed token: a fed one is in `history` already, so its text must go out too.
    auto take = [&](llama_token token, bool fed) {
        if (llama_vocab_is_eog(vocab, token)) { result->finish_reason = 0; return false; }
        if (!fed) unfed = token;
        ++result->generated_tokens;
        stopped |= !emit(token);
        if (stopped && !fed) throw Cancelled();
        return result->generated_tokens < max_tokens;
    };
    if (s->speculative) {
        llama_token anchor = sample(-1);
        bool more = take(anchor, false);
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
                dp.n_past = position;
                dp.id_last = anchor;
                dp.prompt = &history;
                dp.result = &draft;
                const auto draft_start = Clock::now();
                common_speculative_draft(s->speculative.get());
                llama_synchronize(s->draft_context.get());
                // Remove noise tokens before injecting real target features at
                // those same M-RoPE positions (official app/server sequence).
                trim_memory(s, s->draft_context.get(), position);
                check_cancelled(s->control);
                result->draft_seconds += seconds(draft_start);
                // The fork uses its configured width rather than dp.n_max.
                // Bound both verification storage and remaining output budget.
                const size_t draft_limit = static_cast<size_t>(std::min(s->config.draft_max, remaining - 1));
                if (draft.size() > draft_limit) draft.resize(draft_limit);
                if (draft.empty() && strict_draft)
                    throw std::runtime_error("DSpark produced no draft tokens; refusing silent target-only fallback.");
            }
            verify.value.n_tokens = 0;
            verify.add(anchor, position, true);
            for (size_t i = 0; i < draft.size(); ++i) verify.add(draft[i], position + 1 + i, true);
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
            position += 1 + accepted;
            trim_memory(s, s->context.get(), position);
            trim_memory(s, s->draft_context.get(), position);
            if (!draft.empty()) common_speculative_accept(s->speculative.get(), 0, static_cast<uint16_t>(accepted));
            history.push_back(anchor);
            history.insert(history.end(), draft.begin(), draft.begin() + accepted);
            unfed = LLAMA_TOKEN_NULL;
            if (!draft.empty()) ++result->speculation_rounds;
            result->draft_tokens += static_cast<int32_t>(draft.size());
            result->accepted_draft_tokens += accepted;
            result->decoded_tokens += 1 + accepted;
            result->decode_seconds += seconds(decode_start);
            // The accepted drafts are fed; the correction is not.
            for (size_t i = 0; i < ids.size(); ++i) { if (!(more = take(ids[i], i < static_cast<size_t>(accepted)))) break; }
            if (stopped) throw Cancelled();
            anchor = ids.back();
        }
    } else for (int i = 0; i < max_tokens; ++i) {
        check_cancelled(s->control);
        const llama_token token = sample(-1);
        // The last sampled token needs no forward pass. Count evaluated tokens
        // separately so short outputs cannot inflate the decode rate.
        if (!take(token, false)) break;
        const auto decode_start = Clock::now();
        llama_token input_token = token;
        evaluate(s, llama_batch_get_one(&input_token, 1), false);
        history.push_back(token);
        unfed = LLAMA_TOKEN_NULL;
        result->decode_seconds += seconds(decode_start);
        ++result->decoded_tokens;
    }
}

// Tokenizes `formatted` (with one image marker) and `bitmap`, and evaluates every chunk.
// Returns the next position.
static llama_pos prefill_image(minicpm_session * s, const std::string & formatted, const mtmd_bitmap * bitmap,
                               int32_t max_tokens, minicpm_result * result) {
    check_cancelled(s->control);
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    if (!chunks) throw std::bad_alloc();
    mtmd_input_text input{formatted.c_str(), true, true};
    const mtmd_bitmap * bitmaps[] = {bitmap};
    if (mtmd_tokenize(s->vision.get(), chunks.get(), &input, bitmaps, 1) != 0)
        throw std::runtime_error("MiniCPM image/prompt tokenization failed.");
    const auto prompt_tokens = mtmd_helper_get_n_tokens(chunks.get());
    const auto positions = mtmd_helper_get_n_pos(chunks.get());
    if (positions < 1 || static_cast<uint64_t>(positions) + max_tokens > llama_n_ctx(s->context.get()))
        throw std::runtime_error("Prompt and output exceed the 4096-token context. Shorten the prompt or output limit.");
    result->prompt_tokens = static_cast<int32_t>(prompt_tokens);
    llama_pos next_position = 0;
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
    return next_position;
}

// Forgets the ledger, the pending token and the checkpoints (the cache itself is left as it is).
static void forget(minicpm_session * s) {
    s->ledger.clear();
    s->pending = LLAMA_TOKEN_NULL;
    s->logits_ready = false;
    s->dirty = false;
    s->foreign = false;
    for (auto & checkpoint : s->checkpoints) checkpoint = Checkpoint();
}

static void clear_cache(minicpm_session * s) {
    llama_memory_clear(llama_get_memory(s->context.get()), true);
    if (s->draft_context) llama_memory_clear(llama_get_memory(s->draft_context.get()), true);
    forget(s);
}

static void settle(minicpm_session * s) {
    llama_synchronize(s->context.get());
    if (s->draft_context) llama_synchronize(s->draft_context.get());
}

// Runs `body`, mapping its outcome to a status: 0 ok, 1 error, 2 cancelled. A failure while the
// run is cancelled is a cancellation: a stopped graph can fail in ways a finished one can't.
template <typename Body>
static int32_t guarded(minicpm_session * s, char ** error, const char * fallback, Body && body) {
    if (error) *error = nullptr;
    bool cancelled = false;
    try {
        body();
        return 0;
    } catch (const Cancelled &) {
        cancelled = true;
    } catch (const std::exception & e) {
        set_error(error, e.what());
    } catch (...) {
        set_error(error, fallback);
    }
    settle(s);
    if (cancelled || is_cancelled(s->control)) {
        if (error) { std::free(*error); *error = nullptr; }
        set_error(error, "Cancelled.");
        return 2;
    }
    return 1;
}

int32_t minicpm_run(minicpm_session * s, const char * prompt, const char * image_path,
                    int32_t max_tokens, minicpm_result * result, char ** error) {
    if (error) *error = nullptr;
    if (!s || !prompt || !result) { set_error(error, "A loaded session and prompt are required."); return 1; }
    Operation operation(s->control);
    return guarded(s, error, "Unknown MiniCPM inference failure.", [&] {
        check_cancelled(s->control);
        if (max_tokens < 1 || max_tokens > 2048) throw std::runtime_error("Token limit must be between 1 and 2048.");
        if (!*prompt || std::strlen(prompt) > 32768) throw std::runtime_error("Prompt must contain 1–32768 UTF-8 bytes.");
        if (std::strstr(prompt, mtmd_default_marker())) throw std::runtime_error("Prompt cannot contain the reserved image marker.");

        *result = {};
        result->first_text_seconds = -1;
        result->les_enabled = s->les_enabled;
        result->dspark_enabled = bool(s->speculative);
        result->runtime_config = s->config;
        forget(s);
        s->foreign = true;
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
            std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
                mtmd_helper_bitmap_init_from_file(s->vision.get(), image_path), mtmd_bitmap_free);
            if (!bitmap || mtmd_bitmap_is_audio(bitmap.get())) throw std::runtime_error("Could not decode the benchmark image. Use PNG or JPEG.");
            next_position = prefill_image(s, formatted, bitmap.get(), max_tokens, result);
        }
        result->prefill_seconds = seconds(start);
        const llama_vocab * vocab = llama_model_get_vocab(s->model.get());
        std::string output;
        std::vector<char> piece(256);
        auto emit = [&](llama_token token) {
            append_piece(vocab, token, piece, output);
            if (result->first_text_seconds < 0 && has_complete_visible_text(output)) result->first_text_seconds = seconds(start);
            return true;
        };
        llama_token unfed = LLAMA_TOKEN_NULL;
        decode(s, next_position, prompt_history, max_tokens, result, emit, unfed, true);
        result->total_seconds = seconds(start);
        result->text = copy_string(output);
    });
}

// Session API

void minicpm_get_info(const minicpm_session * s, minicpm_info * info) {
    if (!info) return;
    *info = {};
    if (!s) return;
    info->weights_bytes = llama_model_size(s->model.get()) + (s->draft_model ? llama_model_size(s->draft_model.get()) : 0);
    info->checkpoint_bytes = s->checkpoint_bytes;
    info->n_ctx = static_cast<int32_t>(llama_n_ctx(s->context.get()));
    info->n_past = static_cast<int32_t>(s->ledger.size() + (s->pending != LLAMA_TOKEN_NULL ? 1 : 0));
    info->recurrent = s->recurrent;
    info->vision = bool(s->vision);
    info->dspark = bool(s->speculative);
}

int32_t minicpm_tokenize(minicpm_session * s, const char * utf8, bool special, int32_t * out, int32_t cap,
                         int32_t * n_out, char ** error) {
    if (error) *error = nullptr;
    if (n_out) *n_out = 0;
    if (!s || !utf8 || !n_out || cap < 0 || (cap > 0 && !out)) { set_error(error, "Invalid tokenize arguments."); return 1; }
    const auto * vocab = llama_model_get_vocab(s->model.get());
    const size_t length = std::strlen(utf8);
    if (length > 1 << 20) { set_error(error, "Text is too long to tokenize."); return 1; }
    std::vector<llama_token> tokens;
    // No BOS: every piece of a conversation is tokenized on its own.
    auto append = [&](size_t from, size_t to) {
        if (from == to) return;
        const auto at = tokens.size();
        const auto bytes = static_cast<int32_t>(to - from);
        tokens.resize(at + to - from + 1);
        int n = llama_tokenize(vocab, utf8 + from, bytes, tokens.data() + at, bytes + 1, false, special);
        if (n < 0) {
            tokens.resize(at + static_cast<size_t>(-n));
            n = llama_tokenize(vocab, utf8 + from, bytes, tokens.data() + at, -n, false, special);
        }
        if (n < 0) throw std::runtime_error("MiniCPM text tokenization failed.");
        tokens.resize(at + n);
    };
    try {
        size_t from = 0;
        // llama_tokenize skips only CONTROL and UNKNOWN tokens without parse_special: a user-defined
        // tag such as "</think>" in a photo's text would become the real token. Its first character
        // goes on its own, so the tag is plain text, the same way in every render.
        for (size_t at = 0; !special && at < length;) {
            const bool tag = std::any_of(s->user_defined.begin(), s->user_defined.end(), [&](const std::string & text) {
                return length - at >= text.size() && std::memcmp(utf8 + at, text.data(), text.size()) == 0;
            });
            if (!tag) { ++at; continue; }
            const auto lead = static_cast<unsigned char>(utf8[at]);
            const size_t first = std::min(length - at, size_t(lead < 0xC0 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4));
            append(from, at);
            append(at, at + first);
            from = at += first;
        }
        append(from, length);
    } catch (const std::exception & e) {
        set_error(error, e.what());
        return 1;
    }
    *n_out = static_cast<int32_t>(tokens.size());
    if (tokens.size() > static_cast<size_t>(cap)) { set_error(error, "The token buffer is too small."); return 1; }
    std::copy(tokens.begin(), tokens.end(), out);
    return 0;
}

int32_t minicpm_ledger(const minicpm_session * s, int32_t * out, int32_t cap, int32_t * n_out) {
    if (!s || !n_out) return 1;
    const size_t count = s->ledger.size() + (s->pending != LLAMA_TOKEN_NULL ? 1 : 0);
    *n_out = static_cast<int32_t>(count);
    if (!out) return 0;
    if (cap < 0 || static_cast<size_t>(cap) < count) return 1;
    std::copy(s->ledger.begin(), s->ledger.end(), out);
    if (s->pending != LLAMA_TOKEN_NULL) out[s->ledger.size()] = s->pending;
    return 0;
}

static void require_ledger(const minicpm_session * s) {
    if (s->foreign) throw std::runtime_error("The cache holds a benchmark or photo run. Reset the session first.");
}

static void validate_tokens(const minicpm_session * s, const int32_t * tokens, size_t n) {
    const int32_t vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(s->model.get()));
    for (size_t i = 0; i < n; ++i)
        if (tokens[i] < 0 || tokens[i] >= vocabulary_size) throw std::runtime_error("Token is outside the vocabulary.");
}

// Feeds `tokens` after the ledger in microbatches (DSpark extracts each one's layer inputs),
// committing each to the ledger once it is in the cache. `first_is_pending` clears `pending`
// once the first microbatch, which holds it, is in.
static void feed_tokens(minicpm_session * s, const llama_token * tokens, size_t n, bool first_is_pending) {
    if (n == 0) return;
    if (s->ledger.size() + n > llama_n_ctx(s->context.get()))
        throw std::runtime_error("The conversation exceeds the 4096-token context.");
    const size_t ubatch = llama_n_ubatch(s->context.get());
    Batch batch(static_cast<int>(ubatch));
    for (size_t offset = 0; offset < n; offset += ubatch) {
        const size_t end = std::min(offset + ubatch, n);
        const auto base = static_cast<llama_pos>(s->ledger.size());
        batch.value.n_tokens = 0;
        for (size_t i = offset; i < end; ++i) batch.add(tokens[i], base + static_cast<llama_pos>(i - offset), i + 1 == n);
        s->logits_ready = false;
        evaluate(s, batch.value, true);
        s->ledger.insert(s->ledger.end(), tokens + offset, tokens + end);
        if (first_is_pending) { s->pending = LLAMA_TOKEN_NULL; first_is_pending = false; }
    }
    s->logits_ready = true;
}

// Brings the cache to exactly the first `position` ledger tokens, in llama-server's restore order
// for hybrid models: restore the deepest checkpoint at or before it, drop everything after the
// checkpoint, then re-feed the gap. A pure-attention cache only drops its tail. Also repairs a
// dirty cache (position = ledger length).
static void restore_to(minicpm_session * s, size_t position) {
    if (position > s->ledger.size()) throw std::runtime_error("Rewind position is outside the ledger.");
    if (position == s->ledger.size() && !s->dirty) return;
    auto * memory = llama_get_memory(s->context.get());
    size_t from = position;
    if (s->recurrent) {
        const Checkpoint * best = nullptr;
        for (const auto & checkpoint : s->checkpoints)
            if (checkpoint.position >= 0 && static_cast<size_t>(checkpoint.position) <= position &&
                (!best || checkpoint.position > best->position)) best = &checkpoint;
        from = 0;
        if (best && llama_state_seq_set_data_ext(s->context.get(), best->state.data(), best->state.size(), 0,
                                                 LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == best->state.size())
            from = static_cast<size_t>(best->position);
    }
    if (from == 0 || !llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(from), -1)) {
        llama_memory_clear(memory, true);
        if (s->draft_context) llama_memory_clear(llama_get_memory(s->draft_context.get()), true);
        from = 0;
    } else if (s->draft_context) {
        llama_memory_seq_rm(llama_get_memory(s->draft_context.get()), 0, static_cast<llama_pos>(from), -1);
    }
    const std::vector<llama_token> gap(s->ledger.begin() + static_cast<std::ptrdiff_t>(from),
                                       s->ledger.begin() + static_cast<std::ptrdiff_t>(position));
    s->ledger.resize(from);
    for (auto & checkpoint : s->checkpoints)
        if (checkpoint.position > static_cast<int32_t>(position)) checkpoint = Checkpoint();
    s->dirty = false;
    s->logits_ready = false;
    feed_tokens(s, gap.data(), gap.size(), false);
}

int32_t minicpm_feed(minicpm_session * s, uint64_t run, const int32_t * tokens, int32_t n, char ** error) {
    if (error) *error = nullptr;
    if (!s || n < 0 || (n > 0 && !tokens)) { set_error(error, "Invalid feed arguments."); return 1; }
    Operation operation(s->control, run);
    return guarded(s, error, "Unknown MiniCPM feed failure.", [&] {
        check_cancelled(s->control);
        require_ledger(s);
        validate_tokens(s, tokens, static_cast<size_t>(n));
        if (s->dirty) restore_to(s, s->ledger.size());
        std::vector<llama_token> all;
        const bool pending = s->pending != LLAMA_TOKEN_NULL;
        if (pending) all.push_back(s->pending);
        all.insert(all.end(), tokens, tokens + n);
        feed_tokens(s, all.data(), all.size(), pending);
    });
}

// Delivers decoded text through `fn` in complete UTF-8 pieces.
struct TextStream {
    minicpm_text_fn fn;
    void * user;
    const llama_vocab * vocab;
    minicpm_result * result;
    Clock::time_point start;
    std::string bytes;
    std::vector<char> piece = std::vector<char>(256);

    bool operator()(llama_token token) {
        append_piece(vocab, token, piece, bytes);
        const size_t complete = complete_utf8_prefix(bytes);
        if (complete == 0) return true;
        const std::string text = bytes.substr(0, complete);
        bytes.erase(0, complete);
        if (result->first_text_seconds < 0 && has_complete_visible_text(text)) result->first_text_seconds = seconds(start);
        return !fn || fn(text.data(), static_cast<int32_t>(text.size()), user);
    }

    // Delivers an incomplete UTF-8 tail (a generation cut at the limit or stopped mid-character)
    // as it is, so the delivered text accounts for every token the ledger holds.
    void flush() {
        if (fn && !bytes.empty()) fn(bytes.data(), static_cast<int32_t>(bytes.size()), user);
        bytes.clear();
    }
};

static void begin_result(const minicpm_session * s, minicpm_result * result) {
    *result = {};
    result->first_text_seconds = -1;
    result->les_enabled = s->les_enabled;
    result->dspark_enabled = bool(s->speculative);
    result->runtime_config = s->config;
}

int32_t minicpm_generate(minicpm_session * s, uint64_t run, int32_t max_tokens, minicpm_text_fn fn, void * user,
                         minicpm_result * result, char ** error) {
    if (error) *error = nullptr;
    if (!s || !result) { set_error(error, "A loaded session and result are required."); return 1; }
    Operation operation(s->control, run);
    begin_result(s, result);
    return guarded(s, error, "Unknown MiniCPM generation failure.", [&] {
        check_cancelled(s->control);
        require_ledger(s);
        if (max_tokens < 1 || max_tokens > 2048) throw std::runtime_error("Token limit must be between 1 and 2048.");
        const auto start = Clock::now();
        // Logits for the last ledger token: repair, flush the pending token, or feed the last again.
        if (s->dirty) restore_to(s, s->ledger.size());
        if (s->pending != LLAMA_TOKEN_NULL) {
            const llama_token token = s->pending;
            feed_tokens(s, &token, 1, true);
        } else if (!s->logits_ready) {
            if (s->ledger.empty()) throw std::runtime_error("Feed a prompt before generating.");
            const llama_token last = s->ledger.back();
            restore_to(s, s->ledger.size() - 1);
            feed_tokens(s, &last, 1, false);
        }
        if (s->ledger.size() + max_tokens + s->config.draft_max > llama_n_ctx(s->context.get()))
            throw std::runtime_error("Prompt and output exceed the 4096-token context.");
        result->prompt_tokens = static_cast<int32_t>(s->ledger.size());
        result->prefill_seconds = seconds(start);
        if (s->speculative) common_speculative_begin(s->speculative.get(), 0, s->ledger);
        TextStream stream{fn, user, llama_model_get_vocab(s->model.get()), result, start};
        llama_token unfed = LLAMA_TOKEN_NULL;
        s->logits_ready = false;
        try {
            decode(s, static_cast<llama_pos>(s->ledger.size()), s->ledger, max_tokens, result, stream, unfed, false);
        } catch (...) {
            s->pending = unfed;
            stream.flush();
            throw;
        }
        s->pending = unfed;
        stream.flush();
        result->total_seconds = seconds(start);
    });
}

int32_t minicpm_checkpoint(minicpm_session * s, int32_t slot, char ** error) {
    if (error) *error = nullptr;
    if (!s || slot < 0 || slot > 2) { set_error(error, "Checkpoint slots are 0, 1 and 2."); return 1; }
    Operation operation(s->control);
    return guarded(s, error, "Unknown MiniCPM checkpoint failure.", [&] {
        require_ledger(s);
        if (!s->recurrent) return;
        if (s->dirty || s->pending != LLAMA_TOKEN_NULL)
            throw std::runtime_error("Feed the pending tokens before taking a checkpoint.");
        auto & checkpoint = s->checkpoints[slot];
        const size_t size = llama_state_seq_get_size_ext(s->context.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        checkpoint.state.resize(size);
        if (llama_state_seq_get_data_ext(s->context.get(), checkpoint.state.data(), size, 0,
                                         LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size) {
            checkpoint = Checkpoint();
            throw std::runtime_error("Could not copy the recurrent state.");
        }
        checkpoint.position = static_cast<int32_t>(s->ledger.size());
        s->checkpoint_bytes = size;
    });
}

int32_t minicpm_rewind(minicpm_session * s, uint64_t run, int32_t position, char ** error) {
    if (error) *error = nullptr;
    if (!s) { set_error(error, "A loaded session is required."); return 1; }
    Operation operation(s->control, run);
    return guarded(s, error, "Unknown MiniCPM rewind failure.", [&] {
        check_cancelled(s->control);
        require_ledger(s);
        const size_t visible = s->ledger.size() + (s->pending != LLAMA_TOKEN_NULL ? 1 : 0);
        if (position < 0 || static_cast<size_t>(position) > visible) throw std::runtime_error("Rewind position is outside the ledger.");
        if (static_cast<size_t>(position) == visible) {
            if (s->dirty) restore_to(s, s->ledger.size());
            return;
        }
        s->pending = LLAMA_TOKEN_NULL;
        restore_to(s, static_cast<size_t>(position));
    });
}

// Prefix file: magic, version, token count, tokens, then the target's and the draft's sequence
// state, each as a byte count and bytes (count 0 without a draft).
static constexpr uint32_t prefix_magic = 0x5846'5043; // "CPFX"
static constexpr uint32_t prefix_version = 1;

using File = std::unique_ptr<FILE, decltype(&std::fclose)>;

static void write_state(FILE * file, llama_context * context) {
    uint64_t size = 0;
    std::vector<uint8_t> state;
    if (context) {
        size = llama_state_seq_get_size_ext(context, 0, 0);
        state.resize(size);
        if (llama_state_seq_get_data_ext(context, state.data(), size, 0, 0) != size)
            throw std::runtime_error("Could not copy the sequence state.");
    }
    if (std::fwrite(&size, sizeof(size), 1, file) != 1 || (size && std::fwrite(state.data(), 1, size, file) != size))
        throw std::runtime_error("Could not write the prefix file.");
}

int32_t minicpm_prefix_save(minicpm_session * s, const char * path, char ** error) {
    if (error) *error = nullptr;
    if (!s || !path) { set_error(error, "A loaded session and path are required."); return 1; }
    Operation operation(s->control);
    return guarded(s, error, "Unknown MiniCPM prefix save failure.", [&] {
        require_ledger(s);
        if (s->ledger.empty() || s->dirty || s->pending != LLAMA_TOKEN_NULL)
            throw std::runtime_error("Only a settled, nonempty cache can be saved.");
        File file(std::fopen(path, "wb"), std::fclose);
        if (!file) throw std::runtime_error("Could not create the prefix file.");
        const uint32_t header[] = {prefix_magic, prefix_version, static_cast<uint32_t>(s->ledger.size())};
        if (std::fwrite(header, sizeof(header), 1, file.get()) != 1 ||
            std::fwrite(s->ledger.data(), sizeof(llama_token), s->ledger.size(), file.get()) != s->ledger.size())
            throw std::runtime_error("Could not write the prefix file.");
        write_state(file.get(), s->context.get());
        write_state(file.get(), s->draft_context.get());
        if (std::fflush(file.get()) != 0) throw std::runtime_error("Could not write the prefix file.");
    });
}

// A prefix file that can't describe this cache (another model, setup or format, or damaged), as
// opposed to one that couldn't be read just now: only this makes minicpm_prefix_load return 3.
struct Mismatch : std::runtime_error { using std::runtime_error::runtime_error; };

static std::vector<uint8_t> read_state(FILE * file) {
    uint64_t size = 0;
    if (std::fread(&size, sizeof(size), 1, file) != 1 || size > (uint64_t(1) << 32))
        throw Mismatch("The prefix file is damaged.");
    std::vector<uint8_t> state(size);
    if (size && std::fread(state.data(), 1, size, file) != size) throw Mismatch("The prefix file is damaged.");
    return state;
}

int32_t minicpm_prefix_load(minicpm_session * s, const char * path, const int32_t * expect, int32_t n, char ** error) {
    if (error) *error = nullptr;
    if (!s || !path || n <= 0 || !expect) { set_error(error, "A loaded session, path and expected tokens are required."); return 1; }
    Operation operation(s->control);
    bool mismatch = false;
    const int32_t status = guarded(s, error, "Unknown MiniCPM prefix load failure.", [&] {
        clear_cache(s);
        File file(std::fopen(path, "rb"), std::fclose);
        if (!file) throw std::runtime_error("No saved prefix.");
        try {
            uint32_t header[3] = {};
            if (std::fread(header, sizeof(header), 1, file.get()) != 1 || header[0] != prefix_magic ||
                header[1] != prefix_version || header[2] != static_cast<uint32_t>(n))
                throw Mismatch("The saved prefix doesn't match.");
            std::vector<llama_token> tokens(static_cast<size_t>(n));
            if (std::fread(tokens.data(), sizeof(llama_token), tokens.size(), file.get()) != tokens.size() ||
                !std::equal(tokens.begin(), tokens.end(), expect))
                throw Mismatch("The saved prefix doesn't match.");
            const auto target = read_state(file.get());
            const auto draft = read_state(file.get());
            if (target.empty() || (s->draft_context && draft.empty()))
                throw Mismatch("The saved prefix doesn't match this model's configuration.");
            if (llama_state_seq_set_data_ext(s->context.get(), target.data(), target.size(), 0, 0) != target.size() ||
                (s->draft_context && llama_state_seq_set_data_ext(s->draft_context.get(), draft.data(), draft.size(), 0, 0) != draft.size()))
                throw Mismatch("The saved prefix doesn't match this model's configuration.");
            s->ledger = std::move(tokens);
        } catch (const Mismatch &) {
            mismatch = true;
            throw;
        }
    });
    if (status != 0) clear_cache(s);
    return status == 1 && mismatch ? 3 : status;
}

void minicpm_reset(minicpm_session * s) {
    if (!s) return;
    Operation operation(s->control);
    clear_cache(s);
}

int32_t minicpm_describe_image(minicpm_session * s, uint64_t run, const char * system, const char * instruction,
                               const uint8_t * bytes, size_t length, int32_t max_slices, int32_t max_tokens,
                               minicpm_text_fn fn, void * user, minicpm_result * result, char ** error) {
    if (error) *error = nullptr;
    if (!s || !instruction || !bytes || !length || !result) { set_error(error, "A loaded session, instruction, image and result are required."); return 1; }
    Operation operation(s->control, run);
    begin_result(s, result);
    return guarded(s, error, "Unknown MiniCPM photo reading failure.", [&] {
        check_cancelled(s->control);
        if (!s->vision) throw std::runtime_error("This text-only session has no vision projector.");
        if (s->speculative) throw std::runtime_error("DSpark sessions read text only.");
        if (max_tokens < 1 || max_tokens > 2048) throw std::runtime_error("Token limit must be between 1 and 2048.");
        if (max_slices < 1 || max_slices > 9) throw std::runtime_error("Use 1–9 image slices.");
        const std::string system_text = system ? system : "";
        const std::string instruction_text = instruction;
        if (instruction_text.empty() || instruction_text.size() + system_text.size() > 32768)
            throw std::runtime_error("The instruction must contain 1–32768 UTF-8 bytes.");
        if (instruction_text.find(mtmd_default_marker()) != std::string::npos ||
            system_text.find(mtmd_default_marker()) != std::string::npos)
            throw std::runtime_error("The prompt cannot contain the reserved image marker.");
        clear_cache(s);
        s->foreign = true;
        const auto start = Clock::now();
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
            mtmd_helper_bitmap_init_from_buf(s->vision.get(), bytes, length), mtmd_bitmap_free);
        if (!bitmap || mtmd_bitmap_is_audio(bitmap.get())) throw std::runtime_error("Could not decode the photo. Use JPEG or PNG.");
        std::string formatted = system_text.empty() ? "" : "<|im_start|>system\n" + system_text + "<|im_end|>\n";
        formatted += "<|im_start|>user\n" + std::string(mtmd_default_marker()) + "\n" + instruction_text +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        // minicpm_run reads one overview slice; put that back afterwards.
        struct Slices {
            mtmd_context * vision;
            ~Slices() { mtmd_set_image_max_slice_nums(vision, 1); }
        } slices{s->vision.get()};
        mtmd_set_image_max_slice_nums(s->vision.get(), max_slices);
        const llama_pos next_position = prefill_image(s, formatted, bitmap.get(), max_tokens, result);
        result->prefill_seconds = seconds(start);
        TextStream stream{fn, user, llama_model_get_vocab(s->model.get()), result, start};
        llama_tokens history;
        llama_token unfed = LLAMA_TOKEN_NULL;
        try {
            decode(s, next_position, history, max_tokens, result, stream, unfed, false);
        } catch (...) {
            stream.flush();
            throw;
        }
        stream.flush();
        result->total_seconds = seconds(start);
    });
}
