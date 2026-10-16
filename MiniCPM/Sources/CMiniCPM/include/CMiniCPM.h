#ifndef C_MINICPM_H
#define C_MINICPM_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct minicpm_control minicpm_control;
typedef struct minicpm_session minicpm_session;

typedef struct minicpm_runtime_config {
    int32_t checkpoint_layer_interval; // Positive interval; cancellation still checks every scheduler node.
    int32_t draft_max;                 // 1–3 for the pinned DSpark checkpoint.
    int32_t expert_cache_size;         // Per-layer LES cache, never restricts trained expert routing.
    bool direct_greedy;                // Allocation-free argmax with the upstream greedy tie rule.
    int32_t cpu_threads;
    int32_t io_threads;                // Runtime pool is process-wide: different values require a fresh process.
} minicpm_runtime_config;
minicpm_runtime_config minicpm_default_runtime_config(void);

// Cancellation is latched. A cancelled control is never reused for new work.
minicpm_control * minicpm_control_create(void);
void minicpm_control_free(minicpm_control * control);
void minicpm_cancel(minicpm_control * control);
bool minicpm_is_cancelled(const minicpm_control * control);
bool minicpm_wait_until_idle(minicpm_control * control, double timeout_seconds);

// Error strings and result text belong to the caller. Release with minicpm_string_free.
minicpm_session * minicpm_load(const char * model_path, const char * projector_path,
                              minicpm_control * control, char ** error);
// Projector may be null for text-only runs. Expert manifest enables LES; draft
// additionally enables DSpark. Requested features fail closed if unavailable.
minicpm_session * minicpm_load_configured(const char * model_path, const char * projector_path,
                                         const char * experts_manifest_path, const char * draft_path,
                                         minicpm_control * control, char ** error);
minicpm_session * minicpm_load_with_runtime_config(const char * model_path, const char * projector_path,
                                         const char * experts_manifest_path, const char * draft_path,
                                         minicpm_runtime_config config,
                                         minicpm_control * control, char ** error);
void minicpm_free(minicpm_session * session);

typedef struct minicpm_result {
    char * text;
    int32_t prompt_tokens;       // Includes visual embedding tokens.
    int32_t generated_tokens;    // Sampled non-EOG output tokens.
    int32_t decoded_tokens;      // Committed target tokens evaluated (rejected speculative rows excluded).
    double prefill_seconds;      // Image I/O, tokenize, vision encode and prompt evaluation.
    double decode_seconds;       // Synchronized decoding; DSpark includes draft, verify, injection and rollback.
    double first_text_seconds;   // Run start to first complete, non-whitespace UTF-8 text; -1 if none.
    double total_seconds;
    int32_t finish_reason;       // 0 = EOG, 1 = token limit.
    bool les_enabled;
    bool dspark_enabled;
    int32_t draft_tokens;
    int32_t accepted_draft_tokens;
    int32_t speculation_rounds;
    double draft_seconds;       // Draft generation subset of decode_seconds.
    minicpm_runtime_config runtime_config;
} minicpm_result;

// 0 = success, 1 = error, 2 = cancellation. Always initialize result to zero first.
int32_t minicpm_run(minicpm_session * session, const char * prompt, const char * image_path,
                    int32_t max_tokens, minicpm_result * result, char ** error);
void minicpm_string_free(char * value);

// Session API for streaming chat with KV reuse and for photo reading.
// Status: 0 ok, 1 error, 2 cancelled. minicpm_run keeps its signature and output for benchmark comparability.
//
// The ledger is the exact tokens the cache holds, at positions 0..<n. Its last token may be
// "pending": the final token a generation emitted without feeding it (the token at the limit, or
// one whose decode was cancelled). The next feed feeds it first. Every token in the ledger has had
// its text delivered (an incomplete UTF-8 tail too, as raw bytes, at the end of a generation), so
// the ledger always matches the reply text that was delivered. A cancelled decode can leave a hybrid
// model's recurrent state ahead of the ledger; the next call restores the nearest checkpoint and
// re-feeds the gap first.
// minicpm_run and minicpm_describe_image use the cache without a ledger: reset or load a prefix
// before feeding again.
uint64_t minicpm_begin_run(minicpm_control *);                 // per-run id
void     minicpm_cancel_run(minicpm_control *, uint64_t run);  // cancels only that run; minicpm_cancel stays latched
bool     minicpm_run_is_cancelled(minicpm_control *, uint64_t run); // that run, or the control (latched), was cancelled
typedef bool (*minicpm_text_fn)(const char *utf8, int32_t len, void *user); // complete UTF-8 only (but a generation's last call); false = stop
typedef struct { uint64_t weights_bytes, checkpoint_bytes; int32_t n_ctx, n_past; bool recurrent, vision, dspark; } minicpm_info;
void    minicpm_get_info(const minicpm_session *, minicpm_info *);       // weights_bytes = llama_model_size (mapped); n_past = ledger length
// With special false, text stays text: a user-defined tag (<think>, <tool_call>…), which
// llama_tokenize would still match, is split after its first character.
int32_t minicpm_tokenize(minicpm_session *, const char *utf8, bool special, int32_t *out, int32_t cap, int32_t *n_out, char **err);
int32_t minicpm_ledger(const minicpm_session *, int32_t *out, int32_t cap, int32_t *n_out); // out may be null to count
int32_t minicpm_feed(minicpm_session *, uint64_t run, const int32_t *tokens, int32_t n, char **err); // appends to ledger; feeds DSpark
int32_t minicpm_generate(minicpm_session *, uint64_t run, int32_t max_tokens, minicpm_text_fn, void *user,
                         minicpm_result *, char **err);                 // greedy (+exact DSpark); never feeds EOG
int32_t minicpm_checkpoint(minicpm_session *, int32_t slot, char **err); // slot 0..2 at n_past; PARTIAL_ONLY; no-op if !recurrent
int32_t minicpm_rewind(minicpm_session *, uint64_t run, int32_t position, char **err); // llama-server restore order; re-feed gap; truncate ledger
int32_t minicpm_prefix_save(minicpm_session *, const char *path, char **err);           // full sequence state (and the draft's) + ledger
int32_t minicpm_prefix_load(minicpm_session *, const char *path, const int32_t *expect, int32_t n, char **err); // fails unless tokens == expect; 3 = the file doesn't match (delete it)
void    minicpm_reset(minicpm_session *);
int32_t minicpm_describe_image(minicpm_session *, uint64_t run, const char *system, const char *instruction,
                               const uint8_t *bytes, size_t len, int32_t max_slices, int32_t max_tokens,
                               minicpm_text_fn, void *user, minicpm_result *, char **err); // resets first; mtmd_helper_bitmap_init_from_buf

#ifdef __cplusplus
}
#endif
#endif
