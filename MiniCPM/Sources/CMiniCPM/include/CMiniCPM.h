#ifndef C_MINICPM_H
#define C_MINICPM_H
#include <stdbool.h>
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

#ifdef __cplusplus
}
#endif
#endif
